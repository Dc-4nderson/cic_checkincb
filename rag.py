"""RAG helpers: build context from Pinecone results and generate a completion
from an LLM provider (OpenAI REST here)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from data_handling import query_pinecone_by_vector, _openai_embeddings


# -----------------------
# Utilities
# -----------------------

def cosine_similarity(a, b):
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    return float(np.dot(a, b))


def build_context_from_query(pinecone_resp: dict, question: str = None, q_emb=None) -> str:
    """Convert Pinecone results into a ranked context string for LLM input."""
    hits = pinecone_resp.get("matches") or pinecone_resp.get("results") or []
    reranked = []
    for h in hits:
        meta = h.get("metadata") or {}
        checkin = meta.get("checkin") or meta.get("full") or h.get("id") or ""
        timestamp = meta.get("timestamp", "")
        embedding = h.get("values")
        # Prefer Pinecone score if available, else cosine sim if values are returned
        score = h.get("score")
        if score is None and embedding is not None and q_emb is not None:
            score = cosine_similarity(q_emb, embedding)
        reranked.append((score or 0, f"[{timestamp}] {checkin}" if timestamp else checkin))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return "\n---\n".join([r[1] for r in reranked])


def generate_answer_from_context(
    question: str,
    context: str,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 400,
) -> str:
    """Call OpenAI Chat/Completions via REST API using the provided context and question."""
    if not openai_api_key:
        raise ValueError("openai_api_key is required to generate answers")

    import requests

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    system = (
    "You are a helpful assistant answering questions like 'What did I do this month?' or 'What did I do on a specific date?'. "
    "Use the provided context, always include timestamps in 'Month day, year' format (e.g., September 29, 2025) without extra symbols. "
    "Assume the current year is 2025. Summarize all relevant check-ins, not just one. "
    "If only partial info is in context, provide it and note others may be missing. "
    "If nothing is in context, reply: 'I don't know I be Jetting'."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    choices = j.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return j.get("result") or ""


# -----------------------
# RAG main entrypoint
# -----------------------

def rag_answer(
    question: str,
    pinecone_api_url: str,
    pinecone_api_key: str,
    openai_api_key: str,
    topK: int = 5,
) -> Dict[str, Any]:
    """Embed question, query Pinecone (with optional month/year filter), build context, and generate answer."""

    # 1. Embed the question
    q_emb = _openai_embeddings([question], openai_api_key)[0]
    if len(q_emb) != 384:
        raise ValueError(f"Embedding dimension mismatch: got {len(q_emb)}, expected 384")

    # 2. Extract month/year from question
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month = next((m for m in months if m.lower() in question.lower()), None)
    year_match = re.search(r"(20\d{2})", question)
    year = year_match.group(1) if year_match else None

    # 3. Build filter for Pinecone
    pc_filter = None
    if month and year:
        pc_filter = {"month": month, "year": year}
    elif month:
        pc_filter = {"month": month}
    elif year:
        pc_filter = {"year": year}

    # 4. Query Pinecone
    pinecone_resp = query_pinecone_by_vector(
        vector=q_emb,
        topK=topK,
        pinecone_api_url=pinecone_api_url,
        pinecone_api_key=pinecone_api_key,
        include_values=True,
        include_metadata=True,
        filter=pc_filter,
    )

    # 5. Build context string
    context = build_context_from_query(pinecone_resp, question, q_emb)

    # 6. Generate LLM answer
    answer = generate_answer_from_context(question, context, openai_api_key)

    return {"answer": answer, "context": context, "pinecone": pinecone_resp}

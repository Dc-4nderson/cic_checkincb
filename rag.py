"""RAG helpers: build context from Pinecone results and generate a completion
from an LLM provider (OpenAI REST here)."""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from data_handling import load_checkins, query_pinecone_by_vector, _openai_embeddings
import numpy as np


def extract_date_from_question(question: str):
	# Try to extract a date/month/year from the question
	# Supports formats like 'October', '2025', 'October 1', '10/01/2025', etc.
	month_names = [
		'January', 'February', 'March', 'April', 'May', 'June',
		'July', 'August', 'September', 'October', 'November', 'December'
	]
	# Find month name
	for m in month_names:
		if m.lower() in question.lower():
			return m
	# Find year
	year_match = re.search(r'(20\d{2})', question)
	if year_match:
		return year_match.group(1)
	# Find MM/DD/YYYY
	date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', question)
	if date_match:
		try:
			dt = datetime.strptime(date_match.group(0), '%m/%d/%Y')
			return dt.strftime('%B %d, %Y')
		except Exception:
			pass
	return None


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_context_from_query(pinecone_resp: dict, question: str = None, q_emb=None) -> str:
    hits = pinecone_resp.get("matches") or pinecone_resp.get("results")
    parts = []
    date_filter = None
    if question:
        date_filter = extract_date_from_question(question)
    # Rerank by cosine similarity if embeddings are available
    reranked = []
    for h in hits or []:
        meta = h.get("metadata") or {}
        checkin = meta.get("checkin") or h.get("metadata", {}).get("text") or h.get("id")
        timestamp = meta.get("timestamp", "")
        embedding = h.get("values")
        if date_filter:
            if date_filter in timestamp:
                score = cosine_similarity(q_emb, embedding) if embedding and q_emb is not None else 0
                reranked.append((score, f"[{timestamp}] {checkin}"))
        else:
            if checkin:
                score = cosine_similarity(q_emb, embedding) if embedding and q_emb is not None else 0
                if timestamp:
                    reranked.append((score, f"[{timestamp}] {checkin}"))
                else:
                    reranked.append((score, checkin))
    # Sort by score descending
    reranked.sort(reverse=True)
    parts = [x[1] for x in reranked]
    return "\n---\n".join(parts)


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

	import json
	import requests

	url = "https://api.openai.com/v1/chat/completions"
	headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
	system = (
		"You are a helpful assistant. Expect questions like 'What did I do this month?' or 'What did I do on a specific date?'. "
		"Use the provided context to answer, and always include timestamps in 'Month day, year' format (e.g., October 1, 2025) in your answers. "
		"Return a summary of all relevant checkins, not just a single record. "
		"If the answer is not in the context, say you don't know."
	)
	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
	]
	body = {"model": model, "messages": messages, "max_tokens": max_tokens}
	resp = requests.post(url, headers=headers, json=body, timeout=30)
	resp.raise_for_status()
	j = resp.json()
	# extract assistant message
	choices = j.get("choices") or []
	if choices:
		return choices[0].get("message", {}).get("content", "")
	return j.get("result") or ""


def rag_answer(
	question: str,
	pinecone_api_url: str,
	pinecone_api_key: str,
	openai_api_key: str,
	topK: int = 5,
):
	"""High-level RAG: embed question, query Pinecone, build context, generate answer."""
	import os
	embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
	embedding_dim = int(os.getenv("EMBEDDING_DIM", "384"))
	q_emb = _openai_embeddings([question], openai_api_key, model=embedding_model)[0]
	# Optionally trim q_emb to embedding_dim if needed
	if len(q_emb) > embedding_dim:
		q_emb = q_emb[:embedding_dim]
	pinecone_resp = query_pinecone_by_vector(pinecone_api_url, pinecone_api_key, q_emb, topK=topK)
	context = build_context_from_query(pinecone_resp, question, q_emb)
	answer = generate_answer_from_context(question, context, openai_api_key)
	return {"answer": answer, "context": context, "pinecone": pinecone_resp}


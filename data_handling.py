"""Data handling helpers: loading checkins, getting embeddings (OpenAI), and
simple Pinecone HTTP API helpers (upsert/query) using an API URL.

This module purposely avoids heavy dependencies. It uses `requests` when
available and falls back to urllib. It expects API keys/URLs to be passed in or
available via environment variables or a `.env` file.
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

try:
  import requests
except Exception:
  requests = None


def _http_post(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
  """Post JSON to URL. Use requests if available otherwise urllib."""
  if requests:
    r = requests.post(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    return r.json()
  # fallback to urllib
  import urllib.request

  req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={**headers, "Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=30) as resp:
    return json.loads(resp.read().decode("utf-8"))


def load_checkins(path: str = "my_checkins.json") -> List[Dict[str, str]]:
  """Load the checkins JSON file from disk and return list of records."""
  if not os.path.exists(path):
    return []
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def _openai_embeddings(texts: List[str], openai_api_key: str, model: str = None) -> List[List[float]]:
  """Call OpenAI embeddings endpoint via REST and return list of vectors.

  Minimal implementation to avoid extra packages. Requires an OpenAI API key.
  """
  if not openai_api_key:
    raise ValueError("openai_api_key is required for generating embeddings")
  if model is None:
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
  url = "https://api.openai.com/v1/embeddings"
  headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
  body = {"model": model, "input": texts}
  resp = _http_post(url, headers, body)
  # response contains 'data' list with 'embedding'
  return [item["embedding"] for item in resp.get("data", [])]


def _sbert_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
  """Generate embeddings using sentence-transformers (SBERT) which produces 384-dim vectors
  when using models like `all-MiniLM-L6-v2`.

  Raises an informative error if the package is not installed.
  """
  try:
    from sentence_transformers import SentenceTransformer
  except Exception as e:
    raise RuntimeError(
      "SentenceTransformers is required for 384-d embeddings. Install with `pip install sentence-transformers`"
    ) from e

  model = SentenceTransformer(model_name)
  vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
  # convert numpy arrays to lists
  return [v.tolist() for v in vectors]


def get_embeddings(texts: List[str], openai_api_key: Optional[str] = None) -> List[List[float]]:
  """Unified embedding function. Always uses OpenAI embeddings."""
  if not openai_api_key:
    raise ValueError("openai_api_key is required for openai embeddings")
  return _openai_embeddings(texts, openai_api_key)


def upsert_checkins_to_pinecone(
  pinecone_api_url: str,
  pinecone_api_key: str,
  index_name: Optional[str],
  checkins: List[Dict[str, str]],
  openai_api_key: Optional[str] = None,
  batch_size: int = 50,
):
  """Embed checkins and upsert into Pinecone using OpenAI embeddings only."""
  if not checkins:
    return {"upserted": 0}

  texts = [c.get("checkin", "") for c in checkins]
  ids = [str(i) for i in range(len(texts))]

  # generate embeddings using OpenAI
  if not openai_api_key:
    raise ValueError("openai_api_key is required for openai embeddings")
  vectors = _openai_embeddings(texts, openai_api_key)

  # prepare batches for upsert
  upsert_url = pinecone_api_url.rstrip("/") + "/vectors/upsert"
  headers = {"Api-Key": pinecone_api_key}
  if not headers.get("Api-Key"):
    raise ValueError("pinecone_api_key is required")

  total = 0
  for i in range(0, len(vectors), batch_size):
    batch = []
    for j, vec in enumerate(vectors[i : i + batch_size], start=i):
      # include both timestamp and checkin in metadata
      # Convert timestamp to month-day-year format
      raw_ts = checkins[j].get("timestamp", "")
      formatted_ts = raw_ts
      if raw_ts:
        try:
          from datetime import datetime
          dt = datetime.fromisoformat(raw_ts.replace("Z", "").replace("T", " "))
          formatted_ts = dt.strftime("%B %d, %Y")
        except Exception:
          formatted_ts = raw_ts
      metadata = {
        "checkin": checkins[j].get("checkin", ""),
        "timestamp": formatted_ts
      }
      batch.append({"id": ids[j], "values": vec, "metadata": metadata})

    body = {"vectors": batch}
    _http_post(upsert_url, headers, body)
    total += len(batch)

  return {"upserted": total}


def query_pinecone_by_vector(
  pinecone_api_url: str, pinecone_api_key: str, vector: List[float], topK: int = 5
) -> Dict[str, Any]:
  """Query Pinecone by vector using the API URL. Returns the raw JSON response."""
  query_url = pinecone_api_url.rstrip("/") + "/query"
  headers = {"Api-Key": pinecone_api_key}
  body = {"vector": vector, "topK": topK, "includeMetadata": True}
  return _http_post(query_url, headers, body)


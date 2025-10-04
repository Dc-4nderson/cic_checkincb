from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

from pinecone.grpc import PineconeGRPC as Pinecone

def safe_id(text: str) -> str:
    """Truncate to 512 ASCII chars, or use SHA256 hash if longer or non-ASCII."""
    # Ensure ASCII only
    try:
        ascii_text = text.encode("ascii", "ignore").decode("ascii")
    except Exception:
        ascii_text = ""
    if len(ascii_text) > 512 or ascii_text != text:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ascii_text

# ------------------------
# Utility Helpers
# ------------------------

def _http_post(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    """Post JSON to URL. Use requests if available otherwise urllib."""
    if requests:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        return r.json()
    import urllib.request
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_checkins(path: str = "my_checkins.json") -> List[Dict[str, str]]:
    """Load the checkins JSON file from disk and return list of records."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _openai_embeddings(texts: List[str], openai_api_key: str) -> List[List[float]]:
    """Call OpenAI embeddings endpoint via REST and return list of vectors."""
    if not openai_api_key:
        raise ValueError("openai_api_key is required for generating embeddings")

    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    body = {"model": "text-embedding-3-large", "input": texts, "dimensions": 384}
    resp = _http_post(url, headers, body)
    return [item["embedding"] for item in resp.get("data", [])]


def get_embeddings(texts: List[str], openai_api_key: str) -> List[List[float]]:
    """Unified embedding function. Uses OpenAI's large embedding model (384-dim)."""
    return _openai_embeddings(texts, openai_api_key)


def _format_timestamp(raw_ts: str) -> str:
    """Convert ISO timestamp to 'Month Day, Year' if possible."""
    if not raw_ts:
        return raw_ts
    try:
        dt = datetime.fromisoformat(raw_ts.replace("Z", "").replace("T", " "))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return raw_ts


# ------------------------
# Pinecone Operations
# ------------------------

def upsert_checkins_to_pinecone_grpc(
    pinecone_api_key: str,
    index_host: str,
    namespace: str,
    checkins: List[Dict[str, str]],
    openai_api_key: Optional[str] = None,
    batch_size: int = 50,
) -> Dict[str, int]:
    """Embed checkins and upsert into Pinecone using gRPC client."""
    if not checkins:
        return {"upserted": 0}
    if not openai_api_key:
        raise ValueError("openai_api_key is required for openai embeddings")

    # Prepare texts and ids
    texts, ids = [], []
    for c in checkins:
        formatted_ts = _format_timestamp(c.get("timestamp", ""))
        checkin_text = c.get("checkin", "")
        full_text = f"{formatted_ts}: {checkin_text}"
        texts.append(checkin_text)
    ids.append(safe_id(full_text))

    vectors = _openai_embeddings(texts, openai_api_key)

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(host=index_host)

    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = []
        for j, vec in enumerate(vectors[i : i + batch_size], start=i):
            formatted_ts = _format_timestamp(checkins[j].get("timestamp", ""))
            checkin_text = checkins[j].get("checkin", "")
            full_text = f"{formatted_ts}: {checkin_text}"
            month = formatted_ts.split(" ")[0] if formatted_ts else ""
            year = formatted_ts.split(" ")[-1] if formatted_ts else ""
            batch.append({
                "id": safe_id(full_text),
                "values": vec,
                "metadata": {
                    "checkin": checkin_text,
                    "timestamp": formatted_ts,
                    "month": month,
                    "year": year,
                    "full": full_text
                }
            })
        index.upsert(vectors=batch, namespace=namespace)
        total += len(batch)

    return {"upserted": total}


def upsert_checkins_to_pinecone_http(
    pinecone_api_url: str,
    pinecone_api_key: str,
    checkins: List[Dict[str, str]],
    openai_api_key: Optional[str] = None,
    batch_size: int = 50,
) -> Dict[str, int]:
    """Embed checkins and upsert into Pinecone via HTTP API."""
    if not checkins:
        return {"upserted": 0}
    if not openai_api_key:
        raise ValueError("openai_api_key is required for openai embeddings")
    if not pinecone_api_key:
        raise ValueError("pinecone_api_key is required")

    texts, ids = [], []
    for c in checkins:
        formatted_ts = _format_timestamp(c.get("timestamp", ""))
        checkin_text = c.get("checkin", "")
        full_text = f"{formatted_ts}: {checkin_text}"
        texts.append(checkin_text)
    ids.append(safe_id(full_text))

    vectors = _openai_embeddings(texts, openai_api_key)

    upsert_url = pinecone_api_url.rstrip("/") + "/vectors/upsert"
    headers = {"Api-Key": pinecone_api_key}

    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = []
        for j, vec in enumerate(vectors[i : i + batch_size], start=i):
            formatted_ts = _format_timestamp(checkins[j].get("timestamp", ""))
            checkin_text = checkins[j].get("checkin", "")
            full_text = f"{formatted_ts}: {checkin_text}"
            month = formatted_ts.split(" ")[0] if formatted_ts else ""
            year = formatted_ts.split(" ")[-1] if formatted_ts else ""
            batch.append({
                "id": safe_id(full_text),
                "values": vec,
                "metadata": {
                    "checkin": checkin_text,
                    "timestamp": formatted_ts,
                    "month": month,
                    "year": year,
                    "full": full_text
                }
            })

        _http_post(upsert_url, headers, {"vectors": batch})
        total += len(batch)

    return {"upserted": total}


def query_pinecone_by_vector(
    vector: List[float],
    topK: int = 5,
    pinecone_api_url: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    grpc_api_key: Optional[str] = None,
    index_host: Optional[str] = None,
    namespace: str = "",
    include_values: bool = False,
    include_metadata: bool = True,
    filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query Pinecone by vector using the API URL or gRPC. Returns the raw JSON response."""
    body = {
        "vector": vector,
        "topK": topK,
        "includeMetadata": include_metadata,
        "includeValues": include_values,
    }
    if namespace:
        body["namespace"] = namespace
    if filter:
        body["filter"] = filter
    if pinecone_api_url and pinecone_api_key:
        query_url = pinecone_api_url.rstrip("/") + "/query"
        headers = {"Api-Key": pinecone_api_key}
        return _http_post(query_url, headers, body)
    # gRPC fallback (not shown for brevity)
    raise ValueError("Insufficient credentials: need either (pinecone_api_url + pinecone_api_key) or (grpc_api_key + index_host)")

    # -------------------------
    # Try HTTP first
    # -------------------------
    if pinecone_api_url and pinecone_api_key:
        query_url = pinecone_api_url.rstrip("/") + "/query"
        headers = {"Api-Key": pinecone_api_key}
        try:
            return _http_post(query_url, headers, body)
        except Exception as e:
            print(f"[HTTP query failed, falling back to gRPC] {e}")

    # -------------------------
    # Fallback to gRPC
    # -------------------------
    if grpc_api_key and index_host:
        try:
            pc = Pinecone(api_key=grpc_api_key)
            index = pc.Index(host=index_host)
            resp = index.query(vectors=[{"values": vector}], namespace=namespace, top_k=topK, include_metadata=True)
            return resp
        except Exception as e:
            raise RuntimeError(f"Both HTTP and gRPC queries failed: {e}")

    raise ValueError("Insufficient credentials: need either (pinecone_api_url + pinecone_api_key) or (grpc_api_key + index_host)")

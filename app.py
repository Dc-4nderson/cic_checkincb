#import dependencies
#import functions

#init app
#/home
#a database area to show all checkins which get pulled from my JSON file
#rag chatbot pulling context from my pinecone vector database
#/upload checkins 
#coming soon (need slack auth)
"""Small CLI and entrypoint for the checkins RAG project.

Usage examples:
  python app.py upsert    # upsert checkins from my_checkins.json to Pinecone
  python app.py ask "What did I work on yesterday?"

This script reads environment variables from a local `.env` file if present.
Required env vars:
  PINECONE_API_URL - base URL for Pinecone index (e.g. https://<index>.pinecone.io)
  PINECONE_API_KEY - Pinecone API key
  OPENAI_API_KEY - OpenAI API key for embeddings & chat
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
from typing import Optional
try:
	# load .env if available
	load_dotenv()
except Exception:
	# fallback: ignore if python-dotenv not installed
	pass

from data_handling import load_checkins, upsert_checkins_to_pinecone
from rag import rag_answer


def env(name: str, default: Optional[str] = None) -> Optional[str]:
	v = os.getenv(name)
	return v if v is not None else default


def cmd_upsert():
	pinecone_api_url = env("PINECONE_API_URL")
	pinecone_api_key = env("PINECONE_API_KEY")
	openai_api_key = env("OPENAI_API_KEY")
	if not pinecone_api_url or not pinecone_api_key or not openai_api_key:
		print("Missing required env vars: PINECONE_API_URL, PINECONE_API_KEY, OPENAI_API_KEY")
		sys.exit(1)
	checkins = load_checkins("my_checkins.json")
	res = upsert_checkins_to_pinecone(pinecone_api_url, pinecone_api_key, None, checkins, openai_api_key=openai_api_key)
	print("Upsert result:", res)


def cmd_ask(question: str):
	pinecone_api_url = env("PINECONE_API_URL")
	pinecone_api_key = env("PINECONE_API_KEY")
	openai_api_key = env("OPENAI_API_KEY")
	if not pinecone_api_url or not pinecone_api_key or not openai_api_key:
		print("Missing required env vars: PINECONE_API_URL, PINECONE_API_KEY, OPENAI_API_KEY")
		sys.exit(1)
	out = rag_answer(question, pinecone_api_url, pinecone_api_key, openai_api_key)
	print("Answer:\n", out["answer"])
	print("\nContext used:\n", out["context"])


def main(argv=None):
	argv = argv or sys.argv[1:]
	if not argv:
		print("Usage: python app.py <upsert|ask> [question]")
		sys.exit(0)
	cmd = argv[0]
	if cmd == "upsert":
		cmd_upsert()
	elif cmd == "ask":
		if len(argv) < 2:
			print("Usage: python app.py ask \"your question\"")
			sys.exit(1)
		question = argv[1]
		cmd_ask(question)
	else:
		print("Unknown command", cmd)


if __name__ == "__main__":
	main()


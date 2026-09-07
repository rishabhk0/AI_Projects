"""
Grounded LLM Router
--------------------
Same core pipeline as before, extended with:
  - per-document categories, so retrieval (and the UI) can be scoped to a topic
  - ingest_text(), for adding user-uploaded documents at runtime, not just
    the fixed docs/ folder at startup
  - configurable k and refusal threshold per query, instead of fixed globals
"""

import glob
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_store"
DB_PATH = BASE_DIR / "logs.sqlite"

REFUSAL_DISTANCE_THRESHOLD = 1.05
COMPLEXITY_WORD_THRESHOLD = 18
CHEAP_MODEL = "gemini-3.1-flash-lite"
STRONG_MODEL = "gemini-3.5-flash"
MODEL_COST_PER_1K_TOKENS = {CHEAP_MODEL: 0.0001, STRONG_MODEL: 0.0003}

REFUSAL_PHRASES = [
    "don't have enough information", "doesn't contain", "does not contain",
    "not mentioned", "no information", "cannot find", "can't find",
    "not specified", "not provided in", "context does not", "unable to find",
]


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def parse_retry_delay(error_str: str, default: float = 20.0) -> float:
    match = re.search(r"retry in (\d+\.?\d*)s", error_str)
    return float(match.group(1)) + 2 if match else default


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="nimbusstack_docs", embedding_function=embed_fn
    )


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, query TEXT, grounded INTEGER, best_distance REAL,
            model_used TEXT, est_cost_usd REAL, latency_ms REAL, answer TEXT,
            sources TEXT, category TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def ingest_text(filename: str, text: str, category: str, replace_existing: bool = True) -> int:
    """Chunk and add one document's text to the collection, tagged with a
    category so it can be filtered on later. Used both for the built-in
    docs/ folder and for anything a user uploads at runtime."""
    collection = get_chroma_collection()

    if replace_existing:
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    chunks = chunk_text(text)
    if not chunks:
        return 0
    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "category": category} for _ in chunks]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def ingest() -> int:
    """Load every .md file in docs/ as a built-in document, category = the
    filename stem (pricing, security, sla, ...)."""
    total = 0
    doc_paths = sorted(glob.glob(str(DOCS_DIR / "*.md")))
    for filepath in doc_paths:
        filename = os.path.basename(filepath)
        category = Path(filename).stem
        text = Path(filepath).read_text(encoding="utf-8")
        total += ingest_text(filename, text, category)
    print(f"Ingested {total} chunks from {len(doc_paths)} built-in docs.")
    return total


def list_categories() -> list[str]:
    collection = get_chroma_collection()
    data = collection.get(include=["metadatas"])
    cats = sorted({m.get("category", "uploaded") for m in data["metadatas"]})
    return cats


class GroundedRouter:
    def __init__(self, api_key: str | None = None):
        self.collection = get_chroma_collection()
        init_db()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if key:
            genai.configure(api_key=key)
        self.has_llm = bool(key)

    def retrieve(self, query: str, k: int = 3, category: str | None = None):
        where = None
        if category and category != "All":
            where = {"category": category}
        results = self.collection.query(query_texts=[query], n_results=k, where=where)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []
        return list(zip(docs, metas, dists))

    def choose_model(self, query: str, best_distance: float, threshold: float) -> str:
        word_count = len(query.split())
        borderline = best_distance > (threshold * 0.7)
        return STRONG_MODEL if (word_count > COMPLEXITY_WORD_THRESHOLD or borderline) else CHEAP_MODEL

    def generate(self, query: str, context_chunks: list[str], model_name: str,
                 max_retries: int = 4) -> str:
        if not self.has_llm:
            return (
                "[LLM not configured -- set GEMINI_API_KEY] Based on retrieved "
                f"context: {context_chunks[0][:200]}..."
            )
        context = "\n\n---\n\n".join(context_chunks)
        prompt = (
            "Answer the user's question using ONLY the context below. "
            "If the context does not contain the answer, say so explicitly -- "
            "do not use outside knowledge.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        last_error = None
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                last_error = e
                wait = parse_retry_delay(str(e))
                print(f"    retry {attempt + 1}/{max_retries}, waiting {wait:.0f}s: {str(e)[:80]}")
                time.sleep(wait)
        print(f"    generation failed after {max_retries} attempts: {str(last_error)[:150]}")
        return "GENERATION_FAILED"

    def answer(self, query: str, k: int = 3, category: str | None = None,
               distance_threshold: float | None = None) -> dict:
        threshold = distance_threshold if distance_threshold is not None else REFUSAL_DISTANCE_THRESHOLD
        start = time.time()
        retrieved = self.retrieve(query, k=k, category=category)

        if not retrieved:
            grounded, best_distance = False, float("inf")
        else:
            best_distance = min(d for _, _, d in retrieved)
            grounded = best_distance <= threshold

        if not grounded:
            result = {
                "query": query, "grounded": False, "generation_failed": False,
                "answer": "I don't have enough information in the knowledge base to answer that confidently, so I'm not going to guess.",
                "sources": [], "model_used": None, "best_distance": best_distance,
                "est_cost_usd": 0.0, "latency_ms": (time.time() - start) * 1000,
                "category": category or "All",
            }
            self._log(result)
            return result

        context_chunks = [doc for doc, _, _ in retrieved]
        sources = sorted({meta["source"] for _, meta, _ in retrieved})
        model_name = self.choose_model(query, best_distance, threshold)
        answer_text = self.generate(query, context_chunks, model_name)

        generation_failed = answer_text == "GENERATION_FAILED"
        actually_grounded = grounded and not generation_failed and not is_refusal(answer_text)

        approx_tokens = (len(query) + sum(len(c) for c in context_chunks)) / 4
        est_cost = 0.0 if generation_failed else (
            (approx_tokens / 1000) * MODEL_COST_PER_1K_TOKENS.get(model_name, 0.0)
        )

        result = {
            "query": query, "grounded": actually_grounded, "generation_failed": generation_failed,
            "answer": answer_text, "sources": sources, "model_used": model_name,
            "best_distance": best_distance, "est_cost_usd": est_cost,
            "latency_ms": (time.time() - start) * 1000, "category": category or "All",
        }
        self._log(result)
        return result

    def _log(self, result: dict):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO query_log
               (timestamp, query, grounded, best_distance, model_used,
                est_cost_usd, latency_ms, answer, sources, category)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(), result["query"],
                int(bool(result["grounded"])), result["best_distance"],
                result["model_used"], result["est_cost_usd"], result["latency_ms"],
                result["answer"], json.dumps(result["sources"]), result["category"],
            ),
        )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    ingest()

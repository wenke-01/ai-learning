#!/usr/bin/env python
"""Minimal PDF -> RAG pipeline with CLI and HTTP serving.

Steps supported:
1) Scan PDFs recursively.
2) Extract text and save intermediate JSON.
3) Chunk text by approximate token count.
4) Embed chunks (TF-IDF) and store vectors.
5) Retrieve top-k chunks for a question.
6) Build a prompt and optionally call an LLM.
7) Provide CLI and HTTP interfaces.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


@dataclass
class Chunk:
    source_path: str
    chunk_index: int
    text: str


def scan_pdfs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.pdf"))


def extract_pdf_text(pdf_path: Path) -> dict:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return {"path": str(pdf_path), "pages": pages, "text": "\n".join(pages)}


def save_intermediate(data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{Path(data['path']).stem}.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def approximate_tokens(text: str) -> List[str]:
    return text.split()


def chunk_text(text: str, min_tokens: int = 500, max_tokens: int = 800) -> List[str]:
    tokens = approximate_tokens(text)
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) < min_tokens and start != 0:
            chunks[-1] = " ".join(chunks[-1].split() + chunk_tokens)
            break
        chunks.append(" ".join(chunk_tokens))
        start = end
    return chunks


class VectorStore:
    def __init__(self, db_path: Path, vectorizer_path: Path) -> None:
        self.db_path = db_path
        self.vectorizer_path = vectorizer_path

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL
                )
                """
            )
            conn.execute("DELETE FROM chunks")

    def save_vectorizer(self, vectorizer: TfidfVectorizer) -> None:
        with open(self.vectorizer_path, "wb") as handle:
            pickle.dump(vectorizer, handle)

    def load_vectorizer(self) -> TfidfVectorizer:
        with open(self.vectorizer_path, "rb") as handle:
            return pickle.load(handle)

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for chunk, vector in zip(chunks, vectors):
                conn.execute(
                    "INSERT INTO chunks (source_path, chunk_index, text, vector) VALUES (?, ?, ?, ?)",
                    (
                        chunk.source_path,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(vector.tolist()),
                    ),
                )
            conn.commit()

    def load_all(self) -> List[Tuple[int, Chunk, np.ndarray]]:
        rows = []
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute("SELECT id, source_path, chunk_index, text, vector FROM chunks"):
                chunk = Chunk(row[1], row[2], row[3])
                vector = np.array(json.loads(row[4]))
                rows.append((row[0], chunk, vector))
        return rows


def build_chunks(pdf_dir: Path, intermediate_dir: Path, min_tokens: int, max_tokens: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for pdf_path in scan_pdfs(pdf_dir):
        data = extract_pdf_text(pdf_path)
        save_intermediate(data, intermediate_dir)
        for idx, chunk_text_item in enumerate(
            chunk_text(data["text"], min_tokens=min_tokens, max_tokens=max_tokens)
        ):
            chunks.append(Chunk(str(pdf_path), idx, chunk_text_item))
    return chunks


def embed_chunks(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, np.ndarray]:
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors.toarray()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(
    question: str, vectorizer: TfidfVectorizer, store: VectorStore, top_k: int
) -> List[Tuple[Chunk, float]]:
    question_vector = vectorizer.transform([question]).toarray()[0]
    scored = []
    for _, chunk, vector in store.load_all():
        score = cosine_similarity(question_vector, vector)
        scored.append((chunk, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        [
            f"[Source: {chunk.source_path} | Chunk {chunk.chunk_index} | Score {score:.4f}]\n{chunk.text}"
            for chunk, score in retrieved
        ]
    )
    return (
        "You are a helpful assistant. Use the context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def call_openai(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set. Here is the prompt:\n" + prompt

    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": "You answer using the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    import urllib.request

    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def handle_ingest(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.pdf_dir)
    intermediate_dir = Path(args.intermediate_dir)
    db_path = Path(args.db_path)
    vectorizer_path = Path(args.vectorizer_path)

    chunks = build_chunks(pdf_dir, intermediate_dir, args.min_tokens, args.max_tokens)
    if not chunks:
        print("No chunks created. Check PDF directory.")
        return

    vectorizer, vectors = embed_chunks(chunks)
    store = VectorStore(db_path, vectorizer_path)
    store.initialize()
    store.save_vectorizer(vectorizer)
    store.add_chunks(chunks, vectors)
    print(f"Ingested {len(chunks)} chunks from {pdf_dir}")


def handle_query(args: argparse.Namespace) -> None:
    store = VectorStore(Path(args.db_path), Path(args.vectorizer_path))
    vectorizer = store.load_vectorizer()
    retrieved = retrieve_top_k(args.question, vectorizer, store, args.top_k)
    prompt = build_prompt(args.question, retrieved)
    answer = call_openai(prompt) if args.use_llm else prompt
    print(answer)


class QueryHandler(BaseHTTPRequestHandler):
    store: VectorStore
    vectorizer: TfidfVectorizer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.startswith("/health"):
            self._write_json({"status": "ok"})
            return

        if not self.path.startswith("/query"):
            self.send_response(404)
            self.end_headers()
            return

        query = self._parse_query()
        question = query.get("q")
        top_k = int(query.get("k", 3))
        if not question:
            self.send_response(400)
            self.end_headers()
            return

        retrieved = retrieve_top_k(question, self.vectorizer, self.store, top_k)
        prompt = build_prompt(question, retrieved)
        response = {
            "question": question,
            "top_k": top_k,
            "prompt": prompt,
            "chunks": [
                {
                    "source": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "score": score,
                    "text": chunk.text,
                }
                for chunk, score in retrieved
            ],
        }
        self._write_json(response)

    def _parse_query(self) -> dict:
        if "?" not in self.path:
            return {}
        query_string = self.path.split("?", 1)[1]
        params = {}
        for part in query_string.split("&"):
            if "=" in part:
                key, value = part.split("=", 1)
                params[key] = value
        return params

    def _write_json(self, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def handle_serve(args: argparse.Namespace) -> None:
    store = VectorStore(Path(args.db_path), Path(args.vectorizer_path))
    vectorizer = store.load_vectorizer()

    QueryHandler.store = store
    QueryHandler.vectorizer = vectorizer

    server = HTTPServer((args.host, args.port), QueryHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest PDFs into vector store")
    ingest.add_argument("--pdf-dir", required=True, help="Directory containing PDFs")
    ingest.add_argument("--intermediate-dir", default="intermediate", help="Where to store JSON")
    ingest.add_argument("--db-path", default="rag.db", help="SQLite DB path")
    ingest.add_argument("--vectorizer-path", default="vectorizer.pkl", help="Vectorizer path")
    ingest.add_argument("--min-tokens", type=int, default=500, help="Minimum tokens per chunk")
    ingest.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens per chunk")
    ingest.set_defaults(func=handle_ingest)

    query = subparsers.add_parser("query", help="Query the vector store")
    query.add_argument("--db-path", default="rag.db", help="SQLite DB path")
    query.add_argument("--vectorizer-path", default="vectorizer.pkl", help="Vectorizer path")
    query.add_argument("--question", required=True, help="User question")
    query.add_argument("--top-k", type=int, default=3, help="Number of results")
    query.add_argument("--use-llm", action="store_true", help="Call OpenAI API")
    query.set_defaults(func=handle_query)

    serve = subparsers.add_parser("serve", help="Run a simple HTTP server")
    serve.add_argument("--db-path", default="rag.db", help="SQLite DB path")
    serve.add_argument("--vectorizer-path", default="vectorizer.pkl", help="Vectorizer path")
    serve.add_argument("--host", default="127.0.0.1", help="Host address")
    serve.add_argument("--port", type=int, default=8000, help="Port")
    serve.set_defaults(func=handle_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

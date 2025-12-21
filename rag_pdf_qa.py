"""CLI tool for PDF-based Retrieval-Augmented QA using LangChain."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a PDF, build a FAISS index, and answer a query with citations.",
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("query", type=str, help="User question to ask")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for RecursiveCharacterTextSplitter",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for RecursiveCharacterTextSplitter",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature",
    )
    return parser.parse_args()


def load_documents(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks, embedding_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.from_documents(chunks, embeddings)


def run_qa(vector_store, query: str, top_k: int, llm_model: str, temperature: float):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain.invoke({"query": query})


def format_sources(source_documents):
    lines = []
    for idx, doc in enumerate(source_documents, start=1):
        metadata = doc.metadata or {}
        page = metadata.get("page", "unknown")
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = f"{snippet[:237]}..."
        lines.append(f"[{idx}] page={page} | {snippet}")
    return "\n".join(lines) if lines else "No sources returned."


def main() -> None:
    args = parse_args()

    documents = load_documents(args.pdf_path)
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    vector_store = build_vector_store(chunks, args.embedding_model)

    result = run_qa(vector_store, args.query, args.top_k, args.llm_model, args.temperature)
    answer = result.get("result") or ""
    sources = result.get("source_documents", [])

    print("Answer:\n")
    print(answer)
    print("\nCitations:\n")
    print(format_sources(sources))


if __name__ == "__main__":
    main()

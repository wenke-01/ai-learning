"""CLI tool for PDF-based Retrieval-Augmented QA using LangChain."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = ".faiss_index"


def parse_args() -> argparse.Namespace:
    legacy_parser = argparse.ArgumentParser(add_help=False)
    legacy_parser.add_argument("pdf_path", type=Path)
    legacy_parser.add_argument("query", type=str)

    parser = argparse.ArgumentParser(
        description="Index PDF directories and answer questions with citations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    index_parser = subparsers.add_parser(
        "index",
        help="Scan a directory for PDFs and build/update a FAISS index",
    )
    index_parser.add_argument("root_dir", type=Path, help="Directory to scan")
    index_parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(DEFAULT_INDEX_DIR),
        help="Directory to store the FAISS index and manifest",
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for RecursiveCharacterTextSplitter",
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for RecursiveCharacterTextSplitter",
    )
    index_parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace embedding model name",
    )
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from scratch",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Query an existing FAISS index",
    )
    query_parser.add_argument("query", type=str, help="User question to ask")
    query_parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(DEFAULT_INDEX_DIR),
        help="Directory containing the FAISS index and manifest",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve",
    )
    query_parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model name",
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature",
    )
    query_parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace embedding model name",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run validation for add/update/delete index paths",
    )
    validate_parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Chunk size for validation splits",
    )
    validate_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for validation splits",
    )

    args, _ = parser.parse_known_args()
    if args.command is None:
        legacy_args = legacy_parser.parse_args()
        legacy_args.command = "legacy"
        legacy_args.chunk_size = 1000
        legacy_args.chunk_overlap = 150
        legacy_args.embedding_model = DEFAULT_EMBEDDING_MODEL
        legacy_args.top_k = 4
        legacy_args.llm_model = "gpt-4o-mini"
        legacy_args.temperature = 0.2
        return legacy_args

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


def get_embeddings(embedding_model: str):
    if embedding_model == "fake":
        return FakeEmbeddings(size=384)
    return HuggingFaceEmbeddings(model_name=embedding_model)


def build_vector_store(chunks, embedding_model: str, ids: list[str] | None = None):
    embeddings = get_embeddings(embedding_model)
    if ids:
        return FAISS.from_documents(chunks, embeddings, ids=ids)
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


def scan_pdf_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    return sorted(
        [path for path in root_dir.rglob("*.pdf") if path.is_file()],
        key=lambda path: str(path).lower(),
    )


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(index_path: Path) -> dict:
    manifest_path = index_path / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "root_dir": None,
        "embedding_model": None,
        "chunk_size": None,
        "chunk_overlap": None,
        "files": {},
    }


def save_manifest(index_path: Path, manifest: dict) -> None:
    manifest_path = index_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_vector_store(index_path: Path, embedding_model: str) -> FAISS | None:
    index_file = index_path / "index.faiss"
    if not index_file.exists():
        return None
    embeddings = get_embeddings(embedding_model)
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_chunks_for_file(
    pdf_path: Path,
    root_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list, list[str]]:
    documents = load_documents(pdf_path)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    rel_path = pdf_path.relative_to(root_dir).as_posix()
    chunk_ids = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{rel_path}::chunk-{idx}"
        chunk_ids.append(chunk_id)
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["file_path"] = rel_path
        chunk.metadata["chunk_id"] = chunk_id
    return chunks, chunk_ids


def index_directory(
    root_dir: Path,
    index_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    rebuild: bool = False,
) -> None:
    index_path.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(index_path)

    if rebuild:
        manifest = {
            "root_dir": str(root_dir.resolve()),
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "files": {},
        }
    else:
        if manifest.get("root_dir") and manifest.get("root_dir") != str(root_dir.resolve()):
            raise ValueError(
                "Index root_dir does not match. Use --rebuild to reinitialize the index.",
            )
        manifest["root_dir"] = str(root_dir.resolve())
        manifest["embedding_model"] = manifest.get("embedding_model") or embedding_model
        manifest["chunk_size"] = manifest.get("chunk_size") or chunk_size
        manifest["chunk_overlap"] = manifest.get("chunk_overlap") or chunk_overlap
        if manifest.get("embedding_model") != embedding_model:
            raise ValueError(
                "Embedding model mismatch. Use the original model or rebuild the index.",
            )

    pdf_files = scan_pdf_files(root_dir)
    current_files = {pdf.relative_to(root_dir).as_posix(): pdf for pdf in pdf_files}
    manifest_files = manifest.get("files", {})

    removed_files = sorted(set(manifest_files) - set(current_files))
    ids_to_delete: list[str] = []
    for rel_path in removed_files:
        ids_to_delete.extend(manifest_files[rel_path].get("chunk_ids", []))
        manifest_files.pop(rel_path, None)

    changed_or_new: list[tuple[str, Path, str]] = []
    for rel_path, pdf_path in current_files.items():
        file_hash = compute_file_hash(pdf_path)
        previous_hash = manifest_files.get(rel_path, {}).get("hash")
        if previous_hash != file_hash:
            if rel_path in manifest_files:
                ids_to_delete.extend(manifest_files[rel_path].get("chunk_ids", []))
            changed_or_new.append((rel_path, pdf_path, file_hash))

    vector_store = None if rebuild else load_vector_store(index_path, embedding_model)

    if vector_store is None and not changed_or_new and ids_to_delete:
        raise ValueError("No existing index found to delete from. Use --rebuild.")

    if vector_store is None and not changed_or_new:
        print("No PDF changes detected. Index is up to date.")
        return

    if vector_store is None and changed_or_new:
        all_chunks: list = []
        all_ids: list[str] = []
        for rel_path, pdf_path, file_hash in changed_or_new:
            chunks, chunk_ids = build_chunks_for_file(
                pdf_path,
                root_dir,
                chunk_size,
                chunk_overlap,
            )
            all_chunks.extend(chunks)
            all_ids.extend(chunk_ids)
            manifest_files[rel_path] = {"hash": file_hash, "chunk_ids": chunk_ids}
        if not all_chunks:
            raise ValueError("No chunks produced from PDFs. Ensure PDFs contain extractable text.")
        vector_store = build_vector_store(all_chunks, embedding_model, ids=all_ids)
    else:
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
        if changed_or_new:
            documents_to_add = []
            ids_to_add: list[str] = []
            for rel_path, pdf_path, file_hash in changed_or_new:
                chunks, chunk_ids = build_chunks_for_file(
                    pdf_path,
                    root_dir,
                    chunk_size,
                    chunk_overlap,
                )
                documents_to_add.extend(chunks)
                ids_to_add.extend(chunk_ids)
                manifest_files[rel_path] = {"hash": file_hash, "chunk_ids": chunk_ids}
            if documents_to_add:
                vector_store.add_documents(documents_to_add, ids=ids_to_add)

    manifest["files"] = manifest_files
    vector_store.save_local(str(index_path))
    save_manifest(index_path, manifest)

    print("Index update summary:")
    print(f"- PDFs scanned: {len(pdf_files)}")
    print(f"- Added/updated: {len(changed_or_new)}")
    print(f"- Removed: {len(removed_files)}")


def query_index(
    index_path: Path,
    embedding_model: str,
    query: str,
    top_k: int,
    llm_model: str,
    temperature: float,
) -> None:
    vector_store = load_vector_store(index_path, embedding_model)
    if vector_store is None:
        raise FileNotFoundError("Index not found. Run the index command first.")
    result = run_qa(vector_store, query, top_k, llm_model, temperature)
    answer = result.get("result") or ""
    sources = result.get("source_documents", [])

    print("Answer:\n")
    print(answer)
    print("\nCitations:\n")
    print(format_sources(sources))


def write_blank_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


def validate_incremental_updates(chunk_size: int, chunk_overlap: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir) / "pdfs"
        index_path = Path(tmpdir) / "index"
        root_dir.mkdir(parents=True)

        pdf_a = root_dir / "a.pdf"
        pdf_b = root_dir / "b.pdf"
        write_blank_pdf(pdf_a)
        write_blank_pdf(pdf_b)

        index_directory(
            root_dir,
            index_path,
            chunk_size,
            chunk_overlap,
            embedding_model="fake",
            rebuild=True,
        )
        manifest = load_manifest(index_path)
        assert len(manifest["files"]) == 2, "Expected two PDFs indexed."
        hash_a = manifest["files"]["a.pdf"]["hash"]

        write_blank_pdf(pdf_a)
        index_directory(
            root_dir,
            index_path,
            chunk_size,
            chunk_overlap,
            embedding_model="fake",
            rebuild=False,
        )
        manifest = load_manifest(index_path)
        assert manifest["files"]["a.pdf"]["hash"] != hash_a, "Expected PDF a.pdf to update."

        pdf_b.unlink()
        index_directory(
            root_dir,
            index_path,
            chunk_size,
            chunk_overlap,
            embedding_model="fake",
            rebuild=False,
        )
        manifest = load_manifest(index_path)
        assert "b.pdf" not in manifest["files"], "Expected b.pdf to be removed."

        print("Validation completed: add/update/delete paths are working.")


def main() -> None:
    args = parse_args()

    if args.command == "legacy":
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
        return

    if args.command == "index":
        index_directory(
            args.root_dir,
            args.index_path,
            args.chunk_size,
            args.chunk_overlap,
            args.embedding_model,
            rebuild=args.rebuild,
        )
        return

    if args.command == "query":
        query_index(
            args.index_path,
            args.embedding_model,
            args.query,
            args.top_k,
            args.llm_model,
            args.temperature,
        )
        return

    if args.command == "validate":
        validate_incremental_updates(args.chunk_size, args.chunk_overlap)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

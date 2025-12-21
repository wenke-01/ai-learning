# ai-learning
My AI learning journey - Notes, guides, and experiments with AI tools

## Hello World

A tiny Python script is included to print the classic greeting.

Run it with:

```bash
python hello_world.py
```

## PDF RAG CLI

`rag_pdf_qa.py` can index a directory of PDFs into a FAISS index and answer
questions with citations. It uses `PyPDFLoader`, `RecursiveCharacterTextSplitter`,
`FAISS`, and `RetrievalQA`.

```bash
export OPENAI_API_KEY=your_key
# Build/update an index from a directory of PDFs
python rag_pdf_qa.py index path/to/pdfs --index-path .faiss_index

# Query the existing index
python rag_pdf_qa.py query "你的问题" --index-path .faiss_index
```

Legacy single-file usage remains supported:

```bash
python rag_pdf_qa.py path/to/file.pdf "你的问题"
```

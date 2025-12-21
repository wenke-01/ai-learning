# ai-learning
My AI learning journey - Notes, guides, and experiments with AI tools

## Hello World

A tiny Python script is included to print the classic greeting.

Run it with:

```bash
python hello_world.py
```

## PDF RAG CLI

`rag_pdf_qa.py` loads a PDF, splits it into chunks, builds a FAISS index, and answers
questions with citations. It uses `PyPDFLoader`, `RecursiveCharacterTextSplitter`,
`FAISS`, and `RetrievalQA`.

```bash
export OPENAI_API_KEY=your_key
python rag_pdf_qa.py path/to/file.pdf "你的问题"
```

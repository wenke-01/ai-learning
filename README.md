# ai-learning
My AI learning journey - Notes, guides, and experiments with AI tools

## Hello World

A tiny Python script is included to print the classic greeting.

Run it with:

```bash
python hello_world.py
```

## PDF RAG Pipeline

This repo includes a minimal PDF → RAG pipeline with CLI and HTTP serving.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Ingest PDFs

```bash
python rag_pipeline.py ingest --pdf-dir /path/to/pdfs --intermediate-dir intermediate
```

This will:
1. Recursively scan PDFs.
2. Extract text into JSON.
3. Chunk text by approximate token count.
4. Build TF-IDF vectors and store them in SQLite.

### Query from CLI

```bash
python rag_pipeline.py query --question "What is this document about?"
```

Add `--use-llm` to call the OpenAI API (requires `OPENAI_API_KEY`).

### Run HTTP server

```bash
python rag_pipeline.py serve --host 127.0.0.1 --port 8000
```

Then query:

```bash
curl "http://127.0.0.1:8000/query?q=What%20is%20this%20document%20about&k=3"
```

### Run with Docker

Build the image:

```bash
docker build -t pdf-rag .
```

Ingest PDFs (mount a local folder of PDFs and a working directory for outputs):

```bash
docker run --rm \
  -v /path/to/pdfs:/data/pdfs \
  -v /path/to/workdir:/data/work \
  pdf-rag ingest --pdf-dir /data/pdfs --intermediate-dir /data/work/intermediate \
  --db-path /data/work/rag.db --vectorizer-path /data/work/vectorizer.pkl
```

Query from CLI:

```bash
docker run --rm \
  -v /path/to/workdir:/data/work \
  pdf-rag query --question "What is this document about?" \
  --db-path /data/work/rag.db --vectorizer-path /data/work/vectorizer.pkl
```

Serve HTTP API:

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/workdir:/data/work \
  pdf-rag serve --host 0.0.0.0 --port 8000 \
  --db-path /data/work/rag.db --vectorizer-path /data/work/vectorizer.pkl
```

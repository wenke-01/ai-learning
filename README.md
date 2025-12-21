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
export OPENAI_API_KEY="k-proj-p9Tv6hrdW5nfqkdjtbJ1nwSBiPG3pAR-RDLNDO4FO5KQ2oCliJVnYBzqYLHdk3_DAsjlQwg-TrT3BlbkFJgHTINRsrBvqTkk3K3w2sw1TR0h_MRPDFztE75lSAPI_N9wMZh4fvxBPNj0wLT_gn7xvPbLl44A"
python rag_pdf_qa.py ./奇思妙想计算机家.pdf "莱斯利"
```

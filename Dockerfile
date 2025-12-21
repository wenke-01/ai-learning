FROM python:3.11-alpine

WORKDIR /app

RUN apk add --no-cache build-base

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY rag_pipeline.py /app/rag_pipeline.py

ENTRYPOINT ["python", "rag_pipeline.py"]

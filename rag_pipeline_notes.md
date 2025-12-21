# RAG Pipeline Preferences

## 1. PDF 解析
- 优先使用 `pypdf` 或 `pdfplumber`。
- 若包含扫描件，搭配 `pytesseract` 或 `easyocr`。

## 2. 文本切分
- 使用 `langchain.text_splitter`（例如 `RecursiveCharacterTextSplitter`）。

## 3. Embedding
- 优先 `OpenAIEmbeddings` 或 `HuggingFaceEmbeddings`。

## 4. 向量库
- 优先 `FAISS`（本地）或 `Chroma`。

## 5. RAG 组件
- 使用 `VectorStoreRetriever` + `RetrievalQA` 或 `ConversationalRetrievalChain`。

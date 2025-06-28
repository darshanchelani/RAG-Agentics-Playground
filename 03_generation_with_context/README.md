# 🧠 Module 03: Generation with Retrieved Context (Basic RAG)

In this module, we complete the RAG loop:
Retrieve the relevant chunks from Qdrant, and pass them to a language model (LLM) to generate an answer.

---

## 🚀 How It Works

1. User asks a query
2. Query → embedding → Qdrant → fetch relevant chunks
3. Chunks + query → passed to LLM via LangChain's RetrievalQA
4. LLM generates factual, grounded response

---

## 📦 Requirements

- OpenAI API key in `.env`
- Qdrant running (Docker)
- Embeddings from Module 01

---

## 🧪 Run it

```bash
pip install -r requirements.txt
python main.py

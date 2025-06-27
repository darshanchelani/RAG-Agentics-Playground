# 🧠 Module 01: Basic Indexing with Qdrant

This module demonstrates the most basic RAG concept: indexing.

We take a text file, chunk it using LangChain, embed each chunk using OpenAI's embeddings,
and store the resulting vectors in a local Qdrant vector store.

---

## 🚀 How It Works

1. Load `sample.txt`
2. Split into small overlapping chunks
3. Generate vector embeddings
4. Store in Qdrant DB (Docker)

---

## 📦 Requirements

- Python 3.8+
- Docker
- OpenAI API Key

---

## ⚙️ Run Qdrant (locally)

```bash
docker run -p 6333:6333 qdrant/qdrant

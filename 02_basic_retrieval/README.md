# 🔍 Module 02: Basic Retrieval from Qdrant

This module shows how to perform **semantic search** using a natural language query
against an indexed collection of text chunks stored in Qdrant.

---

## 🚀 How It Works

1. Connect to Qdrant DB (which holds our embeddings)
2. Accept a user query
3. Convert query to embedding
4. Perform similarity search
5. Return top-k most relevant chunks

---

## 📦 Requirements

- Docker (Qdrant running)
- Python 3.8+
- `.env` file with OpenAI key

---

## 🧪 Run the script

```bash
pip install -r requirements.txt
python main.py

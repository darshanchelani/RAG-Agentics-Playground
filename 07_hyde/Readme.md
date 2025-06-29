# 🧠 Module 07: HyDE – Hypothetical Document Embedding

This module demonstrates HyDE: a novel technique where we generate a **hypothetical answer** to a user query using the LLM and use it to search the vector DB instead of the original query.

---

## 🔁 Workflow

1. Receive user query
2. Generate a realistic, hypothetical answer
3. Embed that answer
4. Search vector store with the embedding
5. Feed retrieved real context + original question to the LLM
6. Get final factual answer

---

## 🚀 Why It Works

| Problem | HyDE Fix |
|--------|----------|
| Queries may lack context | Imagined answer has more clues |
| User query is abstract | Hypothetical answer is specific |
| Embedding is weak | Hypothetical paragraph → better embedding |

---

## ✅ Run the Code

```bash
pip install -r requirements.txt
python main.py

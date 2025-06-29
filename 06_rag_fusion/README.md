# 🏆 Module 06: RAG Fusion (Reranking from Multi-Query)

This module improves upon multi-query RAG by using **vector reranking**:
- Each document chunk retrieved from different query variants is scored
- We select the **most relevant chunks** based on semantic similarity to the original user query

---

## 🔄 Flow

1. Generate query variants via LLM
2. Retrieve top-k results per variant
3. Score each result vs original query (cosine similarity)
4. Select top-N highest scoring chunks
5. Pass top-ranked chunks into LLM

---

## ✅ Benefits

| Feature | Impact |
|--------|--------|
| Reranking | Improves precision of retrieved chunks |
| Deduplication | Prevents repeated info |
| Context Quality | Results in better factual answers |

---
Enter your query: eq (What does Krishna teach about selfless action?)
---
## 🧪 Run it

```bash
pip install -r requirements.txt
python main.py

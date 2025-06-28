import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate

load_dotenv()

# 1. Load LLM and Embeddings
llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Connect to Qdrant
qdrant_client = QdrantClient(url="http://localhost:6333")

qdrant_vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="rag-index-demo",
    embeddings=embeddings,
)

# 3. Base user query
user_query = "What did Krishna say about devotion?"

# 4. Prompt for multi-query generation
prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant. Rephrase the following question in different ways to improve document search.

Original Question: {question}

Return 3 rephrased versions:"""
)

# 5. Generate variants
response = llm.invoke(prompt.format(question=user_query))
rephrased_output = response.content.strip()

query_variants = [line.strip("-• ").strip() for line in rephrased_output.split("\n") if line.strip()]

# 6. Search per variant
retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 2})
all_docs = []

print(f"🔍 Variants of '{user_query}':")
for i, variant in enumerate(query_variants, 1):
    docs = retriever.invoke(variant)
    all_docs.extend(docs)
    print(f"\n{i}. {variant}")
    for d in docs:
        print(f"   → {d.page_content[:80]}...")

# 7. Deduplicate
unique_docs = []
seen = set()
for doc in all_docs:
    if doc.page_content not in seen:
        unique_docs.append(doc)
        seen.add(doc.page_content)

# 8. Prepare final prompt
context = "\n\n".join([doc.page_content for doc in unique_docs[:4]])
final_prompt = f"""Use the context below to answer the user's question.

Context:
{context}

Question: {user_query}
Answer:"""

# 9. Get final answer
answer = llm.invoke(final_prompt)

print("\n" + "=" * 80)
print(f"💬 Final Answer:\n{answer.content}")

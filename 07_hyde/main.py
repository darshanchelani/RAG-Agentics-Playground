import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM and embeddings
llm = ChatOpenAI(temperature=0, api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(api_key=openai_api_key)

# Connect to Qdrant vector store
qdrant_client = QdrantClient(url="http://localhost:6333")
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name="rag-index-demo",
    embedding=embedding_model
)

# Step 1: User query
user_query = "What does the Gita say about the mind and self-control?"

# Step 2: Generate a hypothetical answer
prompt = f"""
You are an expert on Hindu philosophy.

Given the question: "{user_query}"

Generate a short paragraph that might appear in a document answering this question.
Do not mention that this is hypothetical.
"""
hypothetical_answer = llm.invoke(prompt).content.strip()

# Step 3: Get embedding of the generated answer
query_embedding = embedding_model.embed_query(hypothetical_answer)

# Step 4: Search similar documents using the embedding
retrieved_docs = qdrant.similarity_search_by_vector(query_embedding, k=4)

# Step 5: Final answer with context
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
final_prompt = f"""
Use the context below to answer the user's question.

Context:
{context}

Question: {user_query}

Answer:
"""
final_answer = llm.invoke(final_prompt).content.strip()

# Output
print("=" * 80)
print(f"📥 User Query:\n{user_query}\n")
print(f"🤖 Hypothetical Answer (for retrieval):\n{hypothetical_answer}\n")
print("📚 Retrieved Documents:\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i}. {doc.page_content[:100]}...\n{'-'*60}")
print(f"\n💬 Final Answer:\n{final_answer}")

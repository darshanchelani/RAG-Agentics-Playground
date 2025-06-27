import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Connect to the existing Qdrant collection using the new QdrantVectorStore
qdrant = QdrantVectorStore(
    client=QdrantClient(url="http://localhost:6333"),
    collection_name="rag-index-demo",
    embedding=embeddings
)

# Define a query
query = "What is the Bhagavad Gita about?"

# Perform similarity search
results = qdrant.similarity_search(query, k=3)

# Display results
print(f"🔍 Query: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}\n{'-'*60}")

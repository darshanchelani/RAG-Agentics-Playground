import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings  # updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from qdrant_client import QdrantClient

load_dotenv()

# 1. Load document
loader = TextLoader("sample.txt")
docs = loader.load()

# 2. Split document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# 3. Load OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 4. Connect to Qdrant running locally via Docker
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    location="http://localhost:6333",
    collection_name="rag-index-demo"
)

print(f"✅ Indexed {len(chunks)} chunks into Qdrant.")

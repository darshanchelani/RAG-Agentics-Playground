import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

load_dotenv()

# Load OpenAI models
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Qdrant
qdrant = QdrantVectorStore(
    client=QdrantClient(url="http://localhost:6333"),
    collection_name="rag-index-demo",
    embedding=embeddings,
)

# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=qdrant.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Ask a question
query = "Explain the teachings of Krishna in the Bhagavad Gita?"
result = qa_chain.invoke(query)  

# Print result
print(f"💬 Query: {query}\n")
print(f"🧠 Answer:\n{result['result']}\n")

# Show sources
print("📚 Sources used:\n")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"{i}. {doc.page_content}\n{'-'*60}")

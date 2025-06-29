import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# ✅ Connect to Qdrant running via Docker (not local embedded mode)
qdrant_client = QdrantClient(url="http://localhost:6333")

# Connect to existing collection
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name="rag-index-demo",
    embedding=embeddings,
)

# Ask user for input
user_query = input("Enter your query: ")

# Step 1: Prompt to generate variants
variant_prompt = PromptTemplate.from_template("""
Generate 3 different versions of the following question to capture diverse aspects of the information need:
Original Question: {question}
Variations:""")
variant_response = llm.invoke(variant_prompt.format(question=user_query))

# Extract and clean up response
query_variants_text = variant_response.content.strip()
query_variants = [q.strip("- ").strip() for q in query_variants_text.split("\n") if q.strip()]

# Step 2: Search for documents
all_docs = []
for query in query_variants:
    docs = qdrant.similarity_search(query, k=4)
    all_docs.extend(docs)

# Step 3: Deduplicate results
seen = set()
unique_docs = []
for doc in all_docs:
    if doc.page_content not in seen:
        unique_docs.append(doc)
        seen.add(doc.page_content)

# Step 4: Answer prompt
context = "\n\n".join([doc.page_content for doc in unique_docs])
final_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:""")
response = llm.invoke(final_prompt.format(context=context, question=user_query))
print("\nAnswer:\n", response.content.strip())

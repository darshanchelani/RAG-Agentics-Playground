import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient

load_dotenv()

# 🎯 Load models
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# 🔗 Connect to Qdrant
qdrant = QdrantVectorStore(
    client=QdrantClient(url="http://localhost:6333"),
    collection_name="rag-index-demo",
    embedding=embeddings,
)

# ✍️ Custom prompt for query rephrasing
query_template = PromptTemplate(
    input_variables=["user_input"],
    template="""
Rephrase the following question into a search-optimized query
that retrieves factual and relevant information from the Bhagavad Gita:

Question: {user_input}
Optimized Query:"""
)

user_input = "What does Krishna say about doing your duty?"

# 1️⃣ Rephrase the query

rephrased_query = llm.invoke(query_template.format(user_input=user_input)).content.strip()


# 2️⃣ Build RetrievalQA chain with rephrased query
retriever = qdrant.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# 3️⃣ Run the QA chain
result = qa_chain.invoke(rephrased_query)

# 📊 Display results
print(f"🧠 Original Query: {user_input}")
print(f"🛠️ Optimized Query: {rephrased_query}\n")
print(f"💬 Answer:\n{result['result']}\n")

print("📚 Sources Used:")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"{i}. {doc.page_content}\n{'-'*60}")

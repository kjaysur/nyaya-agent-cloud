import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Fetch Credentials
qdrant_url = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", ""))
qdrant_api_key = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))

clean_url = qdrant_url.replace(":6333", "").strip("/") if qdrant_url else ""

client = QdrantClient(url=clean_url, api_key=qdrant_api_key, check_compatibility=False)

vector_db = QdrantVectorStore(
    client=client,
    collection_name="central_acts",
    embedding=embeddings
)

# k=12 allows deep retrieval across large statutory files
retriever = vector_db.as_retriever(search_kwargs={"k": 12})

@tool
def search_statutory_database(query: str) -> str:
    """Search all 849 Indian Central Acts (BNS, BNSS, BSA, POCSO, IT Act, Companies Act, Income Tax Act, etc.) in Qdrant Cloud."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No matching statutory provisions found in the database."
        
        results = []
        for i, doc in enumerate(docs, 1):
            act_title = doc.metadata.get("act_title", "Central Act").replace("_", " ").title()
            source_file = doc.metadata.get("source_file", "Statute")
            results.append(
                f"--- EVIDENCE ITEM [{i}] ---\n"
                f"STATUTE: {act_title} ({source_file})\n"
                f"TEXT:\n{doc.page_content.strip()}\n"
            )
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Database search error: {str(e)}"

# Statutory DB is the sole authority tool
agent_tools = [search_statutory_database]
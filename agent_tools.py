import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

qdrant_url = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", ""))
qdrant_api_key = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))

clean_url = qdrant_url.replace(":6333", "").strip("/") if qdrant_url else ""

client = QdrantClient(url=clean_url, api_key=qdrant_api_key, check_compatibility=False)

vector_db = QdrantVectorStore(
    client=client,
    collection_name="central_acts",
    embedding=embeddings
)

retriever = vector_db.as_retriever(search_kwargs={"k": 12})
_raw_ddg_tool = DuckDuckGoSearchRun()

@tool
def search_bns_statutes(query: str) -> str:
    """Search post-July 2024 Indian laws and Central Acts (BNS, BNSS, BSA, POCSO, IT Act, etc.)."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No matching statutory sections found in Qdrant database."
        
        results = []
        for i, doc in enumerate(docs, 1):
            act_title = doc.metadata.get("act_title", "Unknown Act")
            results.append(f"[{i}] Source: {act_title}\nText: {doc.page_content}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Error querying statutory vector database: {str(e)}"

@tool
def safe_web_search(query: str) -> str:
    """Search the web for supplementary legal information or recent case law."""
    try:
        return _raw_ddg_tool.invoke(query)
    except Exception as e:
        return f"Web search unavailable ({type(e).__name__}). Relying on statutory vector database evidence."

agent_tools = [search_bns_statutes, safe_web_search]
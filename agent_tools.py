import torch
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': device}
)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})
web_search_tool = DuckDuckGoSearchRun()

@tool
def search_bns_statutes(query: str) -> str:
    """Searches ChromaDB vector store for BNS, BNSS, BSA, and Central Acts provisions."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant statutory text found in database."
    return "\n\n".join([f"[Source: {d.metadata.get('source', 'Act')}]: {d.page_content}" for d in docs])

@tool
def search_case_law_precedents(query: str) -> str:
    """Searches live web sources for landmark High Court or Supreme Court judgments."""
    try:
        return web_search_tool.invoke(f"site:indiankanoon.org Supreme Court judgment {query}")
    except Exception as e:
        return f"Web search failed: {str(e)}"

@tool
def verify_bns_section_mapping(offence_name: str) -> str:
    """Deterministic lookup tool to get exact BNS 2023 section numbers and prevent IPC section leakage."""
    mappings = {
        "murder": "BNS Section 103 (Replaces IPC 302)",
        "culpable homicide": "BNS Section 105 (Replaces IPC 304)",
        "attempt to murder": "BNS Section 109 (Replaces IPC 307)",
        "cheating": "BNS Section 318 (Replaces IPC 420)",
        "criminal breach of trust": "BNS Section 316 (Replaces IPC 406)",
        "criminal conspiracy": "BNS Section 61 (Replaces IPC 120B)",
        "common intention": "BNS Section 3(5) (Replaces IPC 34)",
        "grievous hurt": "BNS Section 117 / 118 (Replaces IPC 325 / 326)"
    }
    return mappings.get(offence_name.lower().strip(), "Mapping not found in index; verify against raw BNS text.")

agent_tools = [search_bns_statutes, search_case_law_precedents, verify_bns_section_mapping]
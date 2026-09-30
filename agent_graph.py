import os
import json
import re
from typing import Dict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from agent_state import LegalResearchState
from agent_tools import agent_tools

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0.0)

# Helper function to prevent token limit errors (Groq 7,000 ITPM cap)
def format_facts_for_prompt(retrieved_facts: List[Dict], max_chars_per_item: int = 500, total_max_chars: int = 3500) -> str:
    if not retrieved_facts:
        return "No evidence retrieved."
    
    formatted_items = []
    for idx, item in enumerate(retrieved_facts, 1):
        question = item.get("question", "N/A")
        tool_name = item.get("tool", "unknown")
        data = str(item.get("data", ""))[:max_chars_per_item]  # Truncate each snippet
        formatted_items.append(f"[{idx}] Question: {question}\nTool ({tool_name}): {data}")
    
    combined = "\n\n".join(formatted_items)
    return combined[:total_max_chars]  # Hard cap on overall text length (~800-1000 tokens)


# NODE 1: Planner
def planner_node(state: LegalResearchState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    You are a Lead Legal Research Strategist. Break down the user query into 2 to 3 concise sub-questions required for legal research under post-July 2024 Indian Law (BNS/BNSS/BSA).
    
    Query: {query}
    
    Output ONLY a valid JSON array of strings. Example: ["Sub-question 1", "Sub-question 2"]
    """)
    chain = prompt | llm
    res = chain.invoke({"query": state["user_query"]})
    clean_json = re.sub(r'```json|```', '', res.content).strip()
    try:
        plan = json.loads(clean_json)
    except:
        plan = [state["user_query"]]
    
    return {
        "research_plan": plan,
        "completed_questions": [],
        "retrieved_facts": [],
        "iteration_count": 0,
        "is_satisfied": False
    }


# NODE 2: Dynamic Tool Executor
def executor_node(state: LegalResearchState) -> Dict:
    plan = state["research_plan"]
    completed = state.get("completed_questions", [])
    remaining = [q for q in plan if q not in completed]
    
    if not remaining:
        return {"is_satisfied": True}
        
    current_q = remaining[0]
    llm_with_tools = llm.bind_tools(agent_tools)
    
    response = llm_with_tools.invoke(f"Gather legal facts and exact statutory sections to answer this sub-question: {current_q}")
    
    new_facts = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            selected_tool = next(t for t in agent_tools if t.name == tool_name)
            output = selected_tool.invoke(tool_args)
            new_facts.append({
                "question": current_q,
                "tool": tool_name,
                "data": output
            })
    else:
        new_facts.append({
            "question": current_q,
            "tool": "llm_direct",
            "data": response.content
        })
        
    completed.append(current_q)
    
    return {
        "completed_questions": completed,
        "retrieved_facts": state.get("retrieved_facts", []) + new_facts,
        "iteration_count": state["iteration_count"] + 1
    }


# NODE 3: Critic & Auditor
def critic_node(state: LegalResearchState) -> Dict:
    # Safely trim context for Critic node
    facts_summary = format_facts_for_prompt(state.get("retrieved_facts", []), max_chars_per_item=300, total_max_chars=2000)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Legal Quality Auditor for Indian Law. Review the retrieved evidence:
    
    {facts}
    
    Check for:
    1. Are repealed IPC sections (e.g. 302, 307, 420) cited instead of BNS 2023 sections?
    2. Are there missing core elements needed to answer: "{query}"?
    
    Respond in JSON format:
    {{
        "is_sufficient": true | false,
        "critique_feedback": "Detailed reason if false, or 'Approved' if true."
    }}
    """)
    chain = prompt | llm
    res = chain.invoke({"facts": facts_summary, "query": state["user_query"]})
    clean_json = re.sub(r'```json|```', '', res.content).strip()
    
    try:
        audit = json.loads(clean_json)
    except:
        audit = {"is_sufficient": True, "critique_feedback": "Approved"}
        
    if state["iteration_count"] >= 2:  # Stop after 2 loops to save tokens and prevent rate limits
        audit["is_sufficient"] = True

    return {
        "is_satisfied": audit["is_sufficient"],
        "critique_feedback": audit.get("critique_feedback", "")
    }


# NODE 4: Final Synthesizer
def synthesizer_node(state: LegalResearchState) -> Dict:
    # Safely format and cap facts payload to max 3500 characters (~800 tokens)
    formatted_facts = format_facts_for_prompt(state.get("retrieved_facts", []), max_chars_per_item=600, total_max_chars=3500)
    
    prompt = ChatPromptTemplate.from_template("""
    You are NyayaAgent, an expert Indian Legal AI. Synthesize a formal legal research memo based on the audited facts below.
    
    User Query: {query}
    Audit Feedback: {critique}
    Audited Evidence:
    {facts}
    
    Mandatory Rules:
    - Strictly use BNS 2023, BNSS 2023, and BSA 2023 provisions. Do NOT cite repealed IPC/CrPC sections.
    - Format output with clear headers, statutory tables, and precise penalties.
    
    Conclude with:
    'Disclaimer: This response is generated by an AI research agent for educational purposes and does not constitute formal legal advice.'
    """)
    chain = prompt | llm
    res = chain.invoke({
        "query": state["user_query"],
        "critique": state.get("critique_feedback", "None"),
        "facts": formatted_facts
    })
    
    return {"final_memo": res.content}


# Conditional Routing Logic
def decide_next_step(state: LegalResearchState):
    if state["is_satisfied"]:
        return "synthesizer"
    return "executor"


# Assemble StateGraph
workflow = StateGraph(LegalResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("critic", critic_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "critic")

workflow.add_conditional_edges(
    "critic",
    decide_next_step,
    {
        "synthesizer": "synthesizer",
        "executor": "executor"
    }
)

workflow.add_edge("synthesizer", END)
graph_app = workflow.compile()
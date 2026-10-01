import os
import json
import re
import time
from typing import Dict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from agent_state import LegalResearchState
from agent_tools import agent_tools

load_dotenv()

# Primary LLM for final legal memo synthesis (capped to avoid OTPM overflow)
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0,
    max_tokens=1500,
    max_retries=6,
    request_timeout=60
)

# Fast LLM for planning, tool invocation, and auditing
llm_fast = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.0,
    max_tokens=300,
    max_retries=6,
    request_timeout=30
)
def format_facts_for_prompt(retrieved_facts: List[Dict], max_chars_per_item: int = 800, total_max_chars: int = 4500) -> str:
    if not retrieved_facts:
        return "No evidence retrieved."
    
    formatted_items = []
    for idx, item in enumerate(retrieved_facts, 1):
        question = item.get("question", "N/A")
        tool_name = item.get("tool", "unknown")
        data = str(item.get("data", ""))[:max_chars_per_item]
        formatted_items.append(f"[{idx}] Question: {question}\nTool ({tool_name}): {data}")
    
    combined = "\n\n".join(formatted_items)
    return combined[:total_max_chars]


def planner_node(state: LegalResearchState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Indian Legal Translator & Strategist. Your job is to bridge the gap between layperson terms and formal Indian statutory nomenclature across all 849 Central Acts (BNS, BNSS, BSA, IT Act, Companies Act, POCSO, Income Tax Act, etc.).

    User Query: {query}

    Translate this query into EXACTLY 2 vector search sub-questions:
    1. Canonical Legal Terminology: Translate everyday language into exact statutory terms, act names, or section concepts (e.g., "stolen credit card" -> "identity theft cheating by impersonation Section 66C 66D IT Act"; "rape of a child" -> "aggravated penetrative sexual assault Section 5 Section 6 POCSO BNS Section 65").
    2. Primary Offense & Penalty Search: Search for the core governing statutory provision and punishment terms.

    Output ONLY a valid JSON array of two strings. Example: ["Sub-question 1", "Sub-question 2"]
    """)
    
    chain = prompt | llm_fast
    res = chain.invoke({"query": state["user_query"]})
    clean_json = re.sub(r'```json|```', '', res.content).strip()
    
    try:
        plan = json.loads(clean_json)
    except Exception:
        plan = [state["user_query"], f"statutory provisions and punishment for {state['user_query']}"]
    
    return {
        "research_plan": plan,
        "completed_questions": [],
        "retrieved_facts": [],
        "iteration_count": 0,
        "is_satisfied": False
    }


def executor_node(state: LegalResearchState) -> Dict:
    plan = state["research_plan"]
    completed = state.get("completed_questions", [])
    remaining = [q for q in plan if q not in completed]
    
    if not remaining:
        return {"is_satisfied": True}
        
    current_q = remaining[0]
    llm_with_tools = llm.bind_tools(agent_tools)
    
    time.sleep(1.5)
    
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


def critic_node(state: LegalResearchState) -> Dict:
    time.sleep(1.0)
    facts_summary = format_facts_for_prompt(state.get("retrieved_facts", []), max_chars_per_item=400, total_max_chars=2500)
    
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
        
    if state["iteration_count"] >= 2:
        audit["is_sufficient"] = True

    return {
        "is_satisfied": audit["is_sufficient"],
        "critique_feedback": audit.get("critique_feedback", "")
    }


def synthesizer_node(state: LegalResearchState) -> Dict:
    time.sleep(1.0)
    formatted_facts = format_facts_for_prompt(state.get("retrieved_facts", []), max_chars_per_item=800, total_max_chars=4500)
    
    prompt = ChatPromptTemplate.from_template("""
    You are NyayaAgent, an expert Indian Legal AI. Synthesize a formal legal research memo based STRICTLY on the audited facts below.
    
    User Query: {query}
    Audit Feedback: {critique}
    Audited Evidence:
    {facts}
    
    STRICT COMPLIANCE RULES:
    1. Rely ONLY on the provided Audited Evidence for statutory section numbers and penalty durations.
    2. Do NOT guess or infer section numbers if they are absent from the evidence.
    3. Strictly use BNS 2023, BNSS 2023, BSA 2023, or applicable Central Acts. Do NOT cite repealed IPC/CrPC sections.
    4. Format output with clear headers, statutory tables, and precise penalties.
    
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


def decide_next_step(state: LegalResearchState):
    if state["is_satisfied"]:
        return "synthesizer"
    return "executor"


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
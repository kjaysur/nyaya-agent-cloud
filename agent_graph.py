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

# Primary LLM for final legal memo synthesis (1500 tokens for full memos)
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0,
    max_tokens=1500,
    max_retries=6,
    request_timeout=60
)

# Fast LLM for internal planning and auditing
llm_fast = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.0,
    max_tokens=300,
    max_retries=6,
    request_timeout=30
)

def format_facts_for_prompt(retrieved_facts: List[Dict], max_chars_per_item: int = 1000, total_max_chars: int = 5000) -> str:
    if not retrieved_facts:
        return "No evidence retrieved."
    
    formatted_items = []
    for idx, item in enumerate(retrieved_facts, 1):
        data = str(item.get("data", ""))[:max_chars_per_item]
        formatted_items.append(f"Result [{idx}]:\n{data}")
    
    combined = "\n\n".join(formatted_items)
    return combined[:total_max_chars]


def planner_node(state: LegalResearchState) -> Dict:
    prompt = ChatPromptTemplate.from_template("""
    You are a Lead Legal Research Strategist for Indian Law.
    Analyze the user query and break it down into EXACTLY 2 vector database search queries:
    1. Primary Offense & Penalty Query: Search for the exact statutory section, definition, and punishment provisions.
    2. Special/Interplay Query: Search for any accompanying special provisions, non-obstante clauses, or procedural rules governing this fact pattern.

    User Query: {query}

    Output ONLY a valid JSON array of two search strings. Example: ["Search query 1", "Search query 2"]
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
    llm_with_tools = llm_fast.bind_tools(agent_tools)
    
    time.sleep(1.5)
    
    response = llm_with_tools.invoke(f"Search the statutory database for: {current_q}")
    
    new_facts = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            try:
                selected_tool = next(t for t in agent_tools if t.name == tool_name)
                output = selected_tool.invoke(tool_args)
            except Exception as err:
                output = f"Search error: {str(err)}"
                
            new_facts.append({
                "question": current_q,
                "tool": tool_name,
                "data": output
            })
    else:
        # Fallback to direct search tool invocation if LLM skips tool call
        selected_tool = agent_tools[0]
        output = selected_tool.invoke({"query": current_q})
        new_facts.append({
            "question": current_q,
            "tool": selected_tool.name,
            "data": output
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
    You are a Legal Quality Auditor. Review the retrieved statutory evidence:
    
    {facts}
    
    Query: {query}

    Audit Criteria:
    - Did we retrieve statutory text from active Indian Central Acts?
    - Is the evidence relevant to answering the query?

    Respond in JSON format ONLY:
    {{"is_sufficient": true, "critique_feedback": "Approved"}}
    """)
    chain = prompt | llm_fast
    res = chain.invoke({"facts": facts_summary, "query": state["user_query"]})
    clean_json = re.sub(r'```json|```', '', res.content).strip()
    
    try:
        audit = json.loads(clean_json)
    except Exception:
        audit = {"is_sufficient": True, "critique_feedback": "Approved"}
        
    if state["iteration_count"] >= 2:
        audit["is_sufficient"] = True

    return {
        "is_satisfied": audit.get("is_sufficient", True),
        "critique_feedback": audit.get("critique_feedback", "Approved")
    }


def synthesizer_node(state: LegalResearchState) -> Dict:
    time.sleep(1.0)
    formatted_facts = format_facts_for_prompt(state.get("retrieved_facts", []), max_chars_per_item=1000, total_max_chars=5000)
    
    prompt = ChatPromptTemplate.from_template("""
    You are NyayaAgent, an expert AI Legal Research System specializing in Indian Statutory Law.
    Synthesize a formal legal research memorandum based STRICTLY on the audited statutory evidence provided below.

    User Query: {query}
    
    Audited Statutory Evidence:
    {facts}

    UNIVERSAL LEGAL COMPLIANCE RULES:
    1. STATUTE ACCURACY: Rely strictly on the STATUTE names provided in the evidence headers (e.g. "Bharatiya Nyaya Sanhita, 2023", "Protection of Children from Sexual Offences Act, 2012", "Information Technology Act, 2000"). Never invent or guess Act titles or acronym expansions.
    2. NO REPEALED LAWS: If the query concerns post-July 2024 offenses, prioritize post-reform Acts (BNS, BNSS, BSA) over repealed legacy laws (IPC, CrPC, Evidence Act) unless explicitly asked for historical comparison.
    3. STRICT GROUNDING: State exact section numbers, minimum/maximum terms, and fine provisions ONLY if explicitly present in the evidence. If text in a chunk is incomplete or truncated, state what is known and note the limitation clearly.
    4. STRUCTURE: Format output cleanly with headers: Issue, Relevant Statutory Provisions, Detailed Legal Analysis, and Summary Table.

    Conclude with:
    'Disclaimer: This response is generated by an AI research agent for educational purposes and does not constitute formal legal advice.'
    """)
    chain = prompt | llm
    res = chain.invoke({
        "query": state["user_query"],
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
import streamlit as st
from agent_graph import graph_app

st.set_page_config(page_title="NyayaAgent: Research Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ NyayaAgent: Autonomous Legal Research Agent")
st.caption("Powered by LangGraph, Groq API, and BNS 2023 Vector Store")

user_query = st.chat_input("Ask a legal research question...")

if user_query:
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        status_box = st.status("🚀 Launching Agent Execution Graph...", expanded=True)
        
        initial_state = {"user_query": user_query}
        final_memo = None
        
        for output in graph_app.stream(initial_state):
            for node_name, node_state in output.items():
                if node_name == "planner":
                    status_box.write("📋 **Planner Node:** Deconstructed query into research sub-questions:")
                    st.json(node_state.get("research_plan", []))
                    
                elif node_name == "executor":
                    count = node_state.get('iteration_count', 1)
                    status_box.write(f"🔍 **Executor Node (Loop {count}):** Invoked tools for pending sub-questions.")
                    
                elif node_name == "critic":
                    status_box.write("🧐 **Critic Node:** Auditing retrieved statutory data for IPC leakage...")
                    if node_state.get("is_satisfied"):
                        status_box.write("✅ Audit Passed: Statutory evidence verified.")
                    else:
                        feedback = node_state.get('critique_feedback')
                        status_box.write(f"⚠️ Audit Flagged Issues: {feedback}. Re-routing to Executor...")
                        
                elif node_name == "synthesizer":
                    status_box.update(label="✨ Research Complete!", state="complete", expanded=False)
                    final_memo = node_state.get("final_memo")
        
        if final_memo:
            st.markdown(final_memo)
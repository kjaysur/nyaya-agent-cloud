import streamlit as st
from agent_graph import graph_app

st.set_page_config(page_title="NyayaAgent: Research Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ NyayaAgent: Autonomous Legal Research Agent")

user_query = st.text_input("Enter your legal query:", placeholder="What is the punishment for rape under BNS 2023?")

if st.button("Run Legal Research") and user_query:
    initial_state = {"user_query": user_query}
    final_output = None
    
    with st.status("⚖️ NyayaAgent is analyzing statutes and conducting research...", expanded=False) as status:
        for output in graph_app.stream(initial_state):
            for node_name, node_state in output.items():
                st.write(f"✔️ Processed: **{node_name}**")
                if node_name == "synthesizer" and "final_memo" in node_state:
                    final_output = node_state.get("final_memo")
        
        status.update(label="✅ Legal Research Complete!", state="complete", expanded=False)
    
    if final_output:
        st.markdown("---")
        st.markdown(final_output)
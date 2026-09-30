from typing import TypedDict, List, Dict, Any, Optional

class LegalResearchState(TypedDict):
    user_query: str
    research_plan: List[str]
    completed_questions: List[str]
    retrieved_facts: List[Dict[str, Any]]
    critique_feedback: str
    final_memo: Optional[str]
    iteration_count: int
    is_satisfied: bool
from typing import List

from langgraph.graph import END, START, MessagesState


class AgentState(MessagesState):
    question: str
    answer: str
    web_search: str
    documents: List[str]

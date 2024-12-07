from typing import List

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    question: str
    answer: str
    web_search: str
    documents: List[str]

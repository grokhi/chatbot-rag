from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchRun

if TYPE_CHECKING:
    from backend.src.langgraph.setup import AgentState

web_search_tool = DuckDuckGoSearchRun()


def web_search(state: AgentState):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

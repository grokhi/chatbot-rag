# from __future__ import annotations

# from typing import TYPE_CHECKING

from duckduckgo_search.exceptions import RatelimitException
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults

from backend.src.core.logger import logger

# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]
    documents = state["documents"]

    logger.debug("WEB SEARCH", extra={"question": question})

    try:
        web_search_tool = DuckDuckGoSearchResults(output_format="list")
        docs = web_search_tool.invoke(question)
        key = "snippet"
    except RatelimitException:
        logger.debug("RateLimitError in DuckDuckGo search. Use TavilySearch instead.")
        web_search_tool = TavilySearchResults()
        docs = web_search_tool.invoke(question)
        key = "content"

    web_results = "\n".join([d[key] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

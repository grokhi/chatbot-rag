# from __future__ import annotations

# from typing import TYPE_CHECKING

from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults

from backend.src.core.logger import logger

# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState

web_search_tool = DuckDuckGoSearchResults(output_format="list")


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

    # Web search
    docs = web_search_tool.invoke(question)
    web_results = "\n".join([d["snippet"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

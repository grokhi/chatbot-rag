# from __future__ import annotations

from backend.src.core.logger import logger
from backend.src.handlers.vector_db import VectorDBHandler

# from typing import TYPE_CHECKING


# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    logger.debug(f"RETRIEVE", extra={"question": question})

    # Retrieval
    documents = VectorDBHandler().retriever.invoke(question.content)
    return {"documents": documents, "question": question}

# from __future__ import annotations

from backend.src.core.logger import logger
from backend.src.handlers.vector_db import VectorDBHandler
from backend.src.langgraph.state import AgentState

# from typing import TYPE_CHECKING


# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState


def retrieve(state: AgentState):

    question = state["question"]
    logger.debug("RETRIEVE", extra={"question": question})

    # Retrieval
    documents = VectorDBHandler().retriever.invoke(question.content)
    return {"documents": documents, "question": question}

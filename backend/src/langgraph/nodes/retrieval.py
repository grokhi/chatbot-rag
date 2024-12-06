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
    documents = VectorDBHandler().retriever.invoke(question)
    # rm duplicates
    documents = list({frozenset(d.metadata.items()): d for d in documents}.values())
    return {"documents": documents, "question": question}

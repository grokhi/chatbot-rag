# from __future__ import annotations

from langchain.tools.retriever import create_retriever_tool

from backend.src.core.logger import logger
from backend.src.handlers.vector_db import VectorDBHandler
from backend.src.langgraph.state import AgentState

# from typing import TYPE_CHECKING


# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState


# @tool("retrieve_docs", response_format="content")
# def retriever_tool(query: str):
#     """Retrieve information related to a query."""
#     # retrieved_docs = vector_store.similarity_search(query, k=2)
#     retrieved_docs = retriever.invoke(query)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
#     )
#     return serialized#, retrieved_docs

retriever = VectorDBHandler().retriever

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_documents",
    "Search and return information relevenat to the input question",
)

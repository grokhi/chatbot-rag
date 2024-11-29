# from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, TypedDict, Union

from langchain.chains.llm import LLMChain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langgraph.graph import END, START, MessagesState, StateGraph

from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.handlers.vector_db import VectorDBHandler
from backend.src.langgraph.nodes import (
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)

# def retrieve(state):
#     """
#         Retrieve documents
#     5
#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     print("---RETRIEVE---")
#     question = state["question"]

#     # Retrieval
#     documents = VectorDBHandler().retriever.get_relevant_documents(question)
#     return {"documents": documents, "question": question}


class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


class LangGraphSetup:
    """Setup class for creating and managing the LangGraph workflow."""

    def __init__(self):
        """Initialize the LangGraph setup with necessary components."""
        self.model = LLMHandler()

    def create_workflow(self) -> StateGraph:
        """
        Create and configure the workflow workflow.

        Returns:
            StateGraph: Compiled workflow graph
        """
        # Initialize the state graph
        workflow = StateGraph(AgentState)

        def decide_to_generate(state) -> Literal["transform_query", "generate"]:
            """
            Determines whether to generate an answer, or re-generate a question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Binary decision for next node to call
            """

            print("---ASSESS GRADED DOCUMENTS---")
            _web_search = state["web_search"]

            if "yes" in _web_search.lower():
                # All documents have been filtered check_relevance
                # We will re-generate a new query
                print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
                return "transform_query"
            else:
                # We have relevant documents, so generate answer
                print("---DECISION: GENERATE---")
                return "generate"

        # Define the nodes
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generatae
        workflow.add_node("transform_query", transform_query)  # transform_query
        workflow.add_node("web_search_node", web_search)  # web search

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", decide_to_generate)
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()


def create_graph() -> StateGraph:
    """
    Factory function to create and return a configured LangGraph instance.

    Returns:
        StateGraph: Configured processing graph
    """
    setup = LangGraphSetup()
    return setup.create_workflow()

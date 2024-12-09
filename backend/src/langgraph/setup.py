# from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, TypedDict, Union

import langchain
from langchain.chains.llm import LLMChain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.src.core.config import settings
from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.langgraph.edges.grading import grade_documents
from backend.src.langgraph.nodes import (
    agent,
    generate,
    retriever_tool,
    rewrite,
    web_search,
)

memory = MemorySaver()


if settings.LANGCHAIN_DEBUG:
    langchain.debug = True


class LangGraphSetup:
    """Setup class for creating and managing the LangGraph workflow."""

    def create_workflow(self) -> StateGraph:
        """
        Create and configure the workflow workflow.

        Returns:
            StateGraph: Compiled workflow graph
        """
        # Initialize the state graph
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", agent)
        workflow.add_node("retrieve", ToolNode([retriever_tool]))
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("generate", generate)
        workflow.add_node("web_search", web_search)

        workflow.add_edge(START, "agent")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: "web_search",
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            grade_documents,
        )
        workflow.add_edge("web_search", END)
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        return workflow.compile(checkpointer=memory)


def create_graph() -> StateGraph:
    """
    Factory function to create and return a configured LangGraph instance.

    Returns:
        StateGraph: Configured processing graph
    """
    setup = LangGraphSetup()
    return setup.create_workflow()

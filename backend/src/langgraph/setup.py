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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.langgraph.nodes import (
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)
from backend.src.langgraph.state import AgentState

memory = MemorySaver()


class LangGraphSetup:
    """Setup class for creating and managing the LangGraph workflow."""

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

            logger.debug("ASSESS GRADED DOCUMENTS")
            if state["web_search"] == "Yes":
                logger.debug(
                    "DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY"
                )
                return "transform_query"
            else:
                # We have relevant documents, so generate answer
                logger.debug("DECISION: AT LEAST ONE DOCUMENT IS RELATED TO THE QUESTION, GENERATE")
                return "generate"

        # def agent(state: AgentState):
        #     if isinstance(state["messages"][-1], HumanMessage):
        #         return {"messages": retrieval_grader.invoke()}
        #     if state["web_search"] == "True":
        #         pass

        # LLMHandler().llm.bind_tools([retrieve])
        # Define the nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("web_search_node", web_search)

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", decide_to_generate)
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=memory)


def create_graph() -> StateGraph:
    """
    Factory function to create and return a configured LangGraph instance.

    Returns:
        StateGraph: Configured processing graph
    """
    setup = LangGraphSetup()
    return setup.create_workflow()

import operator
from typing import Annotated, Any, Dict, List, TypedDict, Union

from langchain.chains.llm import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from numpy import outer

from .llm_handler import LLMHandler
from .logger import logger

# from langgraph.tools.vector_search import VectorSearchTool


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    is_relevant: bool = None

    # query: str
    # context: str
    # initial_answer: str | None
    # reasoning: str | None
    final_answer: Dict[str, Any] | None


# Tool to search the knowledge base using RAG
# @tool
# def search_knowledge_base(query: str) -> str:
#     """
#     Search the vector database for relevant knowledge.
#     Args:
#         query (str): The input question.
#     Returns:
#         str: The most relevant context or a fallback message.
#     """
#     vector_tool = VectorSearchTool(db_path="./db/vector_store")
#     results = vector_tool.search(query, top_k=3)
#     return (
#         "\n".join([doc["content"] for doc in results])
#         if results
#         else "No relevant context found in the knowledge base."
#     )


# Tool to search the Internet using DuckDuckGo
# @tool
# def search_internet(query: str) -> str:
#     """
#     Search the Internet for additional information.
#     Args:
#         query (str): The input question.
#     Returns:
#         str: Search results from DuckDuckGo.
#     """
#     internet_tool = DuckDuckGoSearchRun(api_key=None)  # Replace with API key if required
#     return internet_tool.search(query)

search_internet = DuckDuckGoSearchRun()


class LangGraphSetup:
    """Setup class for creating and managing the LangGraph workflow."""

    def __init__(self):
        """Initialize the LangGraph setup with necessary components."""
        self.model = LLMHandler()

        # Define prompt templates
        self.qa_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Answer the following question based on the provided context. 
            If you cannot find the answer in the context, say "I don't have enough information to answer this question."
            
            Context: {context}
            Question: {query}
            
            Answer:""",
        )

        self.reasoning_template = PromptTemplate(
            input_variables=["query", "context", "initial_answer"],
            template="""
            Given the following question and context, provide a step-by-step reasoning process:
            
            Question: {query}
            Context: {context}
            Initial Answer: {initial_answer}
            
            Reasoning steps:""",
        )

    def create_workflow(self) -> StateGraph:
        """
        Create and configure the workflow workflow.

        Returns:
            StateGraph: Compiled workflow graph
        """
        # Initialize the state graph
        workflow = StateGraph(AgentState)

        # Node: Search knowledge base
        # def search_kb_node(state: AgentState) -> AgentState:
        #     """
        #     Search the knowledge base for relevant information.
        #     """
        #     query = state["input"]
        #     kb_context = search_knowledge_base(query)
        #     state["kb_context"] = kb_context
        #     return state

        # Node: Evaluate relevance
        def evaluate_relevance_node(data: dict) -> dict:
            """
            Evaluate the relevance of knowledge base results.
            """
            context = data.get("kb_context", "No relevant context found")
            if "No relevant context found" in context:
                data["is_relevant"] = False
            else:
                data["is_relevant"] = True
            return data

        # Node: Search Internet
        def search_internet_node(data: dict) -> dict:
            """
            Perform Internet search if the knowledge base lacks relevant data.
            """
            if not data.get("is_relevant", False):
                query = data["input"]
                internet_context = search_internet.invoke(query)
                data["intermediate_steps"].append({"search": internet_context})

            return data

        # Node: Generate response
        def generate_response_node(data: dict) -> dict:
            """
            Generate a final response using the LLM.
            """
            query = data["input"]
            context = (
                data.get("kb_context", "")
                if data["is_relevant"]
                else data["intermediate_steps"][-1]
            )
            response = self.model.llm.invoke(
                [HumanMessage(content=f"Context: {context}\nQuestion: {query}")]
            )
            data["final_answer"] = response.content
            return data

        # workflow.add_node("search_knowledge_base", search_kb_node)

        workflow.add_node("evaluate_relevance", evaluate_relevance_node)

        workflow.add_node("search_internet", search_internet_node)

        workflow.add_node("generate_response", generate_response_node)

        # workflow.set_entry_point("search_knowledge_base")
        # workflow.add_edge("search_knowledge_base", "evaluate_relevance")

        workflow.set_entry_point("evaluate_relevance")

        def condition(data):

            if data["is_relevant"] is True:
                return "relevant"
            else:
                return "irrelevant"

        workflow.add_conditional_edges(
            "evaluate_relevance",
            condition,
            {
                "irrelevant": "search_internet",
                "relevant": "generate_response",
                # "final_answer": END,
            },
        )
        workflow.add_edge("search_internet", "generate_response")
        workflow.set_finish_point("generate_response")

        return workflow.compile()

    def _calculate_confidence(self, answer: str, reasoning: str) -> float:
        """
        Calculate confidence score for the response.

        Args:
            answer: Initial answer from QA chain
            reasoning: Reasoning steps from reasoning chain

        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            confidence = 1.0

            # Reduce confidence for uncertainty indicators
            uncertainty_phrases = [
                "I don't have enough information",
                "I'm not sure",
                "unclear",
                "cannot determine",
                "possibly",
                "might be",
            ]

            for phrase in uncertainty_phrases:
                if phrase.lower() in answer.lower():
                    confidence *= 0.5

            # Adjust confidence based on reasoning quality
            if reasoning:
                words = len(reasoning.split())
                if words > 50:  # Detailed reasoning
                    confidence *= 1.2
                elif words < 20:  # Brief reasoning
                    confidence *= 0.8

            # Ensure confidence stays within bounds
            return min(max(confidence, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5


def create_graph() -> StateGraph:
    """
    Factory function to create and return a configured LangGraph instance.

    Returns:
        StateGraph: Configured processing graph
    """
    setup = LangGraphSetup()
    return setup.create_workflow()

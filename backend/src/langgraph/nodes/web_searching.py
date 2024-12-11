from duckduckgo_search.exceptions import RatelimitException
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, chain
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from src.core.logger import logger
from src.handlers import llm_handler
from src.handlers.llm import LLMHandler

web_search_tool = TavilySearchResults()
# web_search_tool = DuckDuckGoSearchResults()


def web_search(state: MessagesState):
    """
    Web search based on the question.

    Args:
        state (messages): The current graph state.

    Returns:
        dict (messages): Generate final answer using web search results.
    """
    messages = state["messages"]

    logger.debug("WEB SEARCH")

    llm = llm_handler.llm
    llm_with_tools = llm.bind_tools([web_search_tool])

    @chain
    def tool_chain(question: str, config: RunnableConfig):
        ai_msg = llm_with_tools.invoke(question, config=config)
        tool_msgs = web_search_tool.batch(ai_msg.tool_calls, config=config)
        return llm_with_tools.invoke([*messages, ai_msg, *tool_msgs], config=config)

    return {
        "messages": tool_chain.invoke(
            [m for m in messages if isinstance(m, HumanMessage)][-1].content
        )
    }

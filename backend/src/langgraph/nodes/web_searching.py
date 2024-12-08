from duckduckgo_search.exceptions import RatelimitException
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, chain
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState

from backend.src.core.logger import logger
from backend.src.handlers import llm_handler
from backend.src.handlers.llm import LLMHandler
from backend.src.langgraph.state import AgentState


def web_search(state: MessagesState):
    """
    Web search based on the question.

    Args:
        state (messages): The current graph state.

    Returns:
        dict (messages): Generate final answer using web search results.
    """
    messages = state["messages"]
    question = [m for m in messages if isinstance(m, HumanMessage)][-1].content

    print("---WEB SEARCH---")

    # model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)
    llm = llm_handler.llm
    web_search_tool = TavilySearchResults()
    llm_with_tools = llm.bind_tools([web_search_tool])

    @chain
    def tool_chain(user_input: str, config: RunnableConfig):
        ai_msg = llm_with_tools.invoke(user_input, config=config)
        tool_msgs = web_search_tool.batch(ai_msg.tool_calls, config=config)
        return llm_with_tools.invoke([ai_msg, *tool_msgs], config=config)

    return {"messages": tool_chain.invoke(question)}

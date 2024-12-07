# from __future__ import annotations

# from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.langgraph.state import AgentState

# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState

# Prompt
system = (
    "You a chat bot question re-writer that transforms an input question"
    "with the context provided with the chat history."
    "If there's no chat history, return improved original question"
    "Answer in one line message."
    "The answer should not contain your reasoning"
    # "When answer, you need to choose from two options: "
    # "1) original question"
    # "2) question contextualized with chat history."
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate a chat history aware question.",
        ),
        MessagesPlaceholder("chat_history"),
    ]
)
llm = LLMHandler().llm
question_rewriter = re_write_prompt | llm | StrOutputParser()


def transform_query(state: AgentState):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    question = state["question"]
    documents = state["documents"]

    logger.debug("TRANSFORM QUERY", extra={"question": question})

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

# from __future__ import annotations

# from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.langgraph.state import AgentState

# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState

prompt = ChatPromptTemplate(
    [
        (
            "system",
            (
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. Do not mention that you have used the provided context. "
                "Use three sentences maximum and keep the answer concise.\n"
                "Question: {question}"
                "\nContext: {context} "
                "\nAnswer:"
            ),
        )
    ]
)

llm = LLMHandler().llm
rag_chain = prompt | llm | StrOutputParser()

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def generate(state: AgentState):
    question = state["question"]
    messages = state["messages"]
    documents = state["documents"]
    logger.debug(f"GENERATE", extra={"question": question})

    res = conversational_rag_chain.invoke(
        {"question": question, "context": documents},
        config={"configurable": {"session_id": "abc123"}},  # constructs a key "abc123" in `store`.
    )
    print(res)
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "answer": generation,
        # "messages": [AIMessage(content=generation)],
    }

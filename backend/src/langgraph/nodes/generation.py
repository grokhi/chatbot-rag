# from __future__ import annotations

# from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler

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

"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
rag_chain = prompt | LLMHandler().llm | StrOutputParser()


def generate(state):
    question = state["question"]
    logger.debug(f"GENERATE", extra={"question": question})
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

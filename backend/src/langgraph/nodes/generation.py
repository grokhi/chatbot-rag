# from __future__ import annotations

# from typing import TYPE_CHECKING

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from backend.src.handlers.llm import LLMHandler

# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState


llm = LLMHandler().get_llm()

prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm | StrOutputParser()


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.src.handlers.llm import LLMHandler

# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Binary 'yes' or 'no' relevance score")


# Fetch the singleton LLM instance
llm = LLMHandler().llm

# Define the prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Document: {document} Question: {question}")]
)
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)


def grade_documents(state):
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"

    for doc in documents:
        result = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if result.binary_score == "yes":
            filtered_docs.append(doc)
        else:
            web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

# from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.langgraph.state import AgentState

# if TYPE_CHECKING:
#     from backend.src.langgraph.setup import AgentState


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Binary 'yes' or 'no' relevance score")


# Define the prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Document: {document} Question: {question}")]
)
llm = LLMHandler().llm
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)

system = """You an attentive question re-writer that considers the context, 
which is a sequence previously asked questions and updates the question
Context: {context}"""
attention_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(system),
        (
            "human",
            "Here is the actual question: \n\n {question} \n Formulate an improved question considering context.",
        ),
    ]
)
question_rewriter_attention = attention_prompt | llm | StrOutputParser()


def grade_documents(state: AgentState):

    question = state["question"]
    documents = state["documents"]

    logger.debug("GRADE DOCUMENTS", extra={"question": question})

    old_human_messages = [x.content for x in state["messages"] if isinstance(x, HumanMessage)][:-1]

    if len(old_human_messages) > 0:
        question = question_rewriter_attention.invoke(
            {"question": question, "context": old_human_messages}
        )

    filtered_docs = []
    web_search = "No"

    for doc in documents:
        result = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if result.binary_score == "yes":
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

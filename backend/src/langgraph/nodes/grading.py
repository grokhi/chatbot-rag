from typing import TYPE_CHECKING

from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field

from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.langgraph.state import AgentState


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Binary 'yes' or 'no' relevance score")


# Define the prompt
system = (
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    # "Dont forget to consider chat history when answering the question."
)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # MessagesPlaceholder("chat_history"),
        ("human", "Document: {document} Question: {question}"),
    ]
)
llm = LLMHandler().llm
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_grader,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def grade_documents(state: AgentState):

    question = state["question"]
    documents = state["documents"]

    logger.debug("GRADE DOCUMENTS", extra={"question": question})

    filtered_docs = []
    web_search = "No"

    for doc in documents:
        # result = conversational_rag_chain.invoke(
        #     {"question": question, "document": doc.page_content},
        #     config={
        #         "configurable": {"session_id": "abc123"}
        #     },  # constructs a key "abc123" in `store`.
        # )
        result = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content, "chat_history": state["messages"]}
        )
        if result.binary_score == "yes":
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

    # old_human_messages = [x.content for x in state["messages"] if isinstance(x, HumanMessage)][:-1]

    # if len(old_human_messages) > 0:
    #     question = question_rewriter_attention.invoke(
    #         {"question": question, "context": old_human_messages}
    #     )

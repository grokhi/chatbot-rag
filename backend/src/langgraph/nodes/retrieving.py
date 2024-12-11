from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from src.core.logger import logger
from src.handlers import vectorstore

retriever = vectorstore.as_retriever()


@tool("retrieve_docs")
def retriever_tool(query: str):
    "Search in vectorstore and return information relevant to the input query"
    # retrieved_docs = vector_store.similarity_search(query, k=2)
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized  # , retrieved_docs


# retriever_tool = create_retriever_tool(
#     retriever,
#     "retrieve_documents",
#     "Search and return information relevenat to the input question",
# )

from backend.src.handlers.vector_db import VectorDBHandler


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = VectorDBHandler().retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

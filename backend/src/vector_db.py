import json

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class VectorDBHandler:
    def __init__(self, collection_name: str, embedding_model=None):
        """
        Initialize the VectorDBHandler.

        Args:
            collection_name (str): Name of the ChromaDB collection.
            embedding_model: Embedding model to use. Defaults to OpenAIEmbeddings.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def load_data_from_json(self, file_path: str) -> list[Document]:
        """
        Load data from a JSON file and process it into LangChain Documents.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            list[Document]: List of processed LangChain Document objects.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        _documents = []
        for item in data:
            content = (
                f"Question (RU): {item['question_text']}\n"
                f"Question (EN): {item.get('question_eng', '')}\n"
                f"Answer: {item['answer_text']}\n"
            )
            metadata = {"uid": item["uid"], "tags": item["tags"], "version": item["RuBQ_version"]}
            _documents.append(Document(page_content=content, metadata=metadata))
        return _documents

    def create_vectorstore(
        self, documents: list[Document], chunk_size: int = 250, chunk_overlap: int = 0
    ):
        """
        Split documents into chunks and create a Chroma vectorstore.

        Args:
            documents (list[Document]): List of LangChain Document objects.
            chunk_size (int): Size of text chunks. Defaults to 250.
            chunk_overlap (int): Overlap between text chunks. Defaults to 0.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splitted_docs = text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=splitted_docs,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        self.retriever = self.vectorstore.as_retriever()

    def query(self, query_text: str, top_k: int = 5) -> list[Document]:
        """
        Query the vectorstore and retrieve relevant documents.

        Args:
            query_text (str): The text query.
            top_k (int): Number of top results to retrieve. Defaults to 5.

        Returns:
            list[Document]: List of retrieved Document objects.
        """
        if not self.retriever:
            raise ValueError("Vectorstore is not initialized. Call `create_vectorstore` first.")
        return self.retriever.get_relevant_documents(query_text, top_k=top_k)

    def get_vectorstore(self):
        """
        Get the underlying vectorstore object for advanced operations.

        Returns:
            Vectorstore object.
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized. Call `create_vectorstore` first.")
        return self.vectorstore


# Initialize the VectorDBHandler
vector_handler = VectorDBHandler(collection_name="qa_chroma")

# Step 1: Load data from JSON file
documents = vector_handler.load_data_from_json("data/RuBQ_2.0_dev.json")

# Step 2: Create the vector store
vector_handler.create_vectorstore(documents)

if __name__ == "__main__":

    # Step 3: Query the vector store
    query = "Which country does Easter Island belong to?"
    results = vector_handler.query(query)

    # Display results
    for result in results:
        print(f"Content: {result.page_content}\nMetadata: {result.metadata}\n")

import json
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# from langchain.embeddings import (
#     GooglePalmEmbeddings,
#     HuggingFaceEmbeddings,
#     OpenAIEmbeddings,
# )
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class VectorDBHandler:
    """
    Singleton class to manage a VectorDBHandler instance.
    """

    _instance = None  # Class-level private instance

    def __new__(
        cls,
        collection_name: Optional[str] = None,
        embedding_model: Optional[object] = None,
        *args,
        **kwargs,
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(collection_name, embedding_model, *args, **kwargs)
        return cls._instance

    def _initialize(self, collection_name: Optional[str], embedding_model: Optional[object]):
        """
        Initialize the VectorDBHandler instance. Runs only once.
        """
        self.collection_name = collection_name
        # self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.embedding_model = embedding_model or GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
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

        # Adjust your existing code
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        filter_complex_metadata(documents)
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

    def get_retriever(self):
        """
        Get the underlying vectorstore object for advanced operations.

        Returns:
            Vectorstore object.
        """
        if not self.retriever:
            raise ValueError("Retriever is not initialized. Call `create_vectorstore` first.")
        return self.retriever


vector_handler = VectorDBHandler(collection_name="qa_chroma")
documents = vector_handler.load_data_from_json("data/RuBQ_2.0_dev.json")
vector_handler.create_vectorstore(documents)

if __name__ == "__main__":

    # Step 3: Query the vector store
    query = "Which country does Easter Island belong to?"
    results = vector_handler.query(query)

    # Display results
    for result in results:
        print(f"Content: {result.page_content}\nMetadata: {result.metadata}\n")

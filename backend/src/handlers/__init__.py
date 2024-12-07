from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from backend.src.handlers.llm import LLMHandler
from backend.src.handlers.vector_db import VectorDBHandler

# initialize singletons
LLMHandler(llm=ChatGroq(model="llama-3.1-70b-versatile", temperature=0))
vector_handler = VectorDBHandler(
    collection_name="qa_chroma",
    embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
)
documents = vector_handler.load_data_from_json("data/datasets/RuBQ_2.0_dev.json")
vector_handler.create_vectorstore(documents)

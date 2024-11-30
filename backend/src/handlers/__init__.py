from backend.src.handlers.llm import LLMHandler
from backend.src.handlers.vector_db import VectorDBHandler

# initialize singletons
LLMHandler(model="llama-3.1-70b-versatile")
vector_handler = VectorDBHandler(collection_name="qa_chroma")
documents = vector_handler.load_data_from_json("data/RuBQ_2.0_dev.json")
vector_handler.create_vectorstore(documents)

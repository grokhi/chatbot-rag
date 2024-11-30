import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain.schema import Document
from pydantic import BaseModel, Field

from backend.src.core.config import settings
from backend.src.core.logger import logger
from backend.src.handlers.llm import LLMHandler
from backend.src.handlers.vector_db import VectorDBHandler
from backend.src.langgraph.setup import create_graph

app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG-based chatbot with document retrieval and LLM integration",
    version="1.0.0",
)

# Initialize components
# vector_db = VectorDBHandler()
llm_handler = LLMHandler()
graph = create_graph()


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str = Field(default="What is the weather in sf?")
    context: Optional[Dict[str, Any]] = {}


class FeedbackRequest(BaseModel):
    query_id: str
    feedback: str


class QueryResponse(BaseModel):
    question: str
    generation: str
    web_search: str
    documents: List[Document]

    class Config:
        arbitrary_types_allowed = True


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Process chat queries using RAG approach

    Args:
        request: QueryRequest containing query and optional context

    Returns:
        Dict containing answer, sources, and confidence score
    """
    try:
        # Log incoming request
        logger.info(f"Received query: {request.query}")

        relevant_docs = []

        augmented_prompt = {
            "question": request.query,
            "generation": "",
            "web_search": "",
            "documents": [],
        }

        response = graph.invoke(augmented_prompt)

        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/feedback")
# async def process_feedback(request: FeedbackRequest) -> Dict[str, str]:
#     """
#     Handle user feedback on responses

#     Args:
#         request: FeedbackRequest containing query_id and feedback

#     Returns:
#         Dict containing status message
#     """
#     try:
#         # Store feedback for future improvements
#         logger.info(f"Received feedback for query {request.query_id}: {request.feedback}")
#         return {"status": "feedback received"}

#     except Exception as e:
#         logger.error(f"Error processing feedback: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/update-knowledge")
# async def update_knowledge_base(
#     document: UploadFile = File(...), metadata: str = Form(default="{}")
# ) -> Dict[str, str]:
#     """
#     Update knowledge base with new documents

#     Args:
#         document: File to be added to the knowledge base
#         metadata: JSON string containing document metadata

#     Returns:
#         Dict containing status message
#     """
#     try:
#         # Validate metadata JSON
#         try:
#             metadata_dict = json.loads(metadata)
#         except json.JSONDecodeError:
#             raise HTTPException(status_code=400, detail="Invalid metadata JSON")

#         # Process and add new document to vector DB
#         vector_db.add_document(document, metadata_dict)
#         return {"status": "knowledge base updated"}

#     except Exception as e:
#         logger.error(f"Error updating knowledge base: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("api:app", host=Config.HOST, port=Config.PORT, reload=Config.DEBUG_MODE)

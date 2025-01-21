from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging
import os
from typing import Optional, Any, Dict, Union
from rag import get_answer
from qdrant import create_collection, upload_documents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables or set defaults
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "/Users/nikolaynechay/Data_Science/RAG/backend/src/prompt_template.txt")

if not TEMPLATE_PATH or not os.path.exists(TEMPLATE_PATH):
    logger.error(f"Invalid TEMPLATE_PATH: {TEMPLATE_PATH}. Please ensure the file exists.")

# Initialize FastAPI application
app = FastAPI(
    title="RAG ChatBot",
    version="1.0.0",
    description="ChatBot API using Retrieve Augmented Generation model",
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    template_path: Optional[str] = TEMPLATE_PATH

class QueryResponse(BaseModel):
    answer: str
    context: Union[str, Any] = None
class UploadRequest(BaseModel):
    url: str
    collection_name: str
# Chatbot endpoint
@app.post("/api/rag", response_class=JSONResponse, response_model=QueryResponse)
async def chatbot_endpoint(payload: QueryRequest) -> QueryResponse:
    """
    Endpoint to handle incoming requests and return responses.
    """
    try:
        logger.info(f"Received RAG query: {payload.question}")

        # Validate the template path
        if not payload.template_path or not os.path.exists(payload.template_path):
            logger.error(f"Template file not found at path: {payload.template_path}")
            raise HTTPException(
                status_code=400, detail=f"Template file not found at path: {payload.template_path}"
            )

        # Call the RAG system to get the answer
        response_data: Dict[str, Any] = get_answer(
            question=payload.question,
            template_path=payload.template_path,
        )

        answer = response_data.get("answer", "")
        context = response_data.get("context", "")

        logger.info(f"Successfully retrieved answer: {answer}")
        return QueryResponse(answer=answer, context=context)

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        raise HTTPException(
            status_code=400, detail=f"Template file not found: {e}"
        )
    except HTTPException as http_ex:
        # Pass HTTP exceptions directly
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while processing your request."
        )

@app.post("/api/upload", response_class=JSONResponse)
async def upload_documents_endpoint(payload: UploadRequest):
    """
    Endpoint to handle uploading documents to the Qdrant database.
    """
    try:
        logger.info(f"Received request to upload documents from URL: {payload.url}")
        logger.info(f"Collection name: {payload.collection_name}")

        # Create collection
        create_collection(payload.collection_name)
        logger.info(f"Collection '{payload.collection_name}' created successfully.")

        # Upload documents
        upload_documents(url=payload.url, collection_name=payload.collection_name)
        logger.info(f"Documents from '{payload.url}' uploaded successfully.")

        return JSONResponse(
            content={"message": f"Documents from '{payload.url}' uploaded successfully to collection '{payload.collection_name}'."},
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading documents: {e}"
        )

import os
import logging
from dotenv import load_dotenv
from typing import Optional
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from qdrant_client import QdrantClient, models
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
ENV_PATH = "/Users/nikolaynechay/Data_Science/RAG/backend/.env"  
load_dotenv(dotenv_path=ENV_PATH)

# Load environment variables
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
COLLECTION_NAME = "Wiki_2"

# Validate environment variables
if not QDRANT_API_KEY or not QDRANT_ENDPOINT:
    raise ValueError("Environment variables QDRANT_API_KEY and QDRANT_ENDPOINT must be set.")

# Initialize Qdrant client and vector store
client = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY,
)

vector_store = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=OpenAIEmbeddings(),
)

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
)

def create_collection(collection_name: str, vector_size: int = 1536) -> None:
    """
    Create a new collection in the Qdrant database.

    Args:
        collection_name (str): The name of the collection to be created.
        vector_size (int): The size of the vectors. Default is 1536.

    Returns:
        None
    """
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Collection '{collection_name}' created successfully!")
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {e}")
        raise

def upload_documents(url: str, collection_name: str) -> None:
    """
    Load documents from a URL, split them into chunks, and upload them to a specified Qdrant collection.

    Args:
        url (str): The URL of the web resource to load.
        collection_name (str): The name of the collection to upload to.

    Returns:
        None
    """
    try:
        # Reinitialize vector_store for the specified collection
        dynamic_vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=OpenAIEmbeddings(),
        )

        # Load and split the documents
        logger.info(f"Loading documents from URL: {url}")
        loader = WebBaseLoader(url)
        documents = loader.load_and_split(text_splitter)

        # Add metadata to documents
        for doc in documents:
            doc.metadata["source_url"] = url

        # Add documents to Qdrant
        logger.info(f"Uploading {len(documents)} documents to collection '{collection_name}'...")
        dynamic_vector_store.add_documents(documents)
        logger.info(f"Documents from URL '{url}' uploaded successfully!")

    except Exception as e:
        logger.exception(f"Error uploading documents from URL '{url}': {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting Qdrant operations...")
        collection_name_var = "Wiki_12"
        create_collection(collection_name=collection_name_var)
        upload_documents(url="https://en.wikipedia.org/wiki/Main_Page", collection_name=collection_name_var)  # Replace with your URL
        logger.info("All operations completed successfully.")
    except Exception as e:
        logger.critical(f"Application failed: {e}")
        exit(1)
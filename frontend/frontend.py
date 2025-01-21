import streamlit as st
from typing import Dict, Optional
import requests
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")

# Routers
CHATING_URL = 'http://localhost:8000/api/rag'
UPLOAD_URL = 'http://localhost:8000/api/upload'
DEFAULT_TEMPLATE_PATH = "/Users/nikolaynechay/Data_Science/RAG/backend/src/prompt_template.txt"

# Page Configuration: Must come before any other Streamlit commands
st.set_page_config(
    page_title='RAG System',
    page_icon="🤖",
    layout='wide',  # Full-width layout
    initial_sidebar_state='auto'
)

client = QdrantClient(
    url=QDRANT_ENDPOINT, 
    api_key=QDRANT_API_KEY
)

def query_backend(question: str, template_path: str = DEFAULT_TEMPLATE_PATH):
    """
    Sends a query to the FastAPI backend and returns the response.
    """
    try:
        payload = {
            'question': question,
            'template_path': template_path
        }
        response = requests.post(CHATING_URL, json=payload, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'Error {response.status_code}: {response.text}'}
    except requests.exceptions.RequestException as e:
        return {'error': f"Connection error: {str(e)}"}

def upload_documents(url: str, collection_name: Optional[str]) -> Dict:
    """
    Sends a URL to the FastAPI backend for document uploading.
    """
    try:
        payload = {
            'url': url,
            'collection_name': collection_name
        }

        response = requests.post(
            url=UPLOAD_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'Error {response.status_code}: {response.text}'}
    except requests.exceptions.RequestException as e:
        return {'error': f"Connection error: {str(e)}"}
    

# def is_collection_name_unique(collection_name: str) -> bool:
#     """
#     Check if the collection name is unique in the Qdrant database.
#     """
#     try:
#         collections = client.get_collections().collections
#         existing_names = [collection.name for collection in collections]
#         return collection_name not in existing_names
#     except Exception as e:
#         st.error(f"Failed to check collection name uniqueness: {e}")
#         return False
        

def main():
    """
    Streamlit application main function that handles UI for:
      1) Chatbot queries
      2) Uploading documents
    """
    st.title('🤖 RAG System Interface')
    st.markdown(
        """
        Welcome to the **RAG System**!  
        Use the tabs below to interact with the chatbot or upload new documents to the Qdrant database.
        """
    )

    # Sidebar settings
    st.sidebar.title('Settings')
    template_path = st.sidebar.text_input(
        'Prompt template path', value=DEFAULT_TEMPLATE_PATH, help='Path to the prompt template file.'
    )
    if not os.path.exists(template_path):
        st.sidebar.warning("The provided template path does not exist. Please update it.")

    # Create tabs
    tab1, tab2 = st.tabs(['ChatBot', 'Document Uploader'])

    # Chatbot tab
    with tab1:
        st.subheader('Chatbot')
        question = st.text_input(
            '💬 Enter your question:', help='Type the question you want to ask the chatbot.'
        )
        submit_button = st.button('Submit')

        if submit_button:
            if question.strip():
                with st.spinner('🤖 Chatbot is thinking...'):
                    response = query_backend(
                        question=question,
                        template_path=template_path
                    )

                    if response.get('error'):
                        st.error(f"An error occurred: {response['error']}")
                    else:
                        st.success('Answer retrieved successfully!', icon='✅')
                        st.subheader('*Answer:*')
                        st.write(response.get('answer', 'No answer found.'))

                        st.subheader('*Context:*')
                        context = response.get('context', 'No context found.')
                        if isinstance(context, str):
                            st.write(context)
                        else:
                            st.json(context)
            else:
                st.warning('Please enter a question to continue.')

    # Document uploader tab
    with tab2:
        st.subheader('Document Uploader')
        upload_url = st.text_input(
            '🔗 Enter the URL for data upload:', help='Provide the URL of the web resource to upload to the Qdrant database.'
        )
        upload_collection_name = st.text_input(
            '📁 Collection Name:', help='Provide the name of the collection to store the uploaded documents.'
        )
        upload_button = st.button('Upload')

        if upload_button:
            if upload_url.strip() and upload_collection_name.strip():
                # TODO: Add logic to verify unique collection names in the database
                with st.spinner('🔄 Uploading data to Qdrant...'):
                    response = upload_documents(
                        url=upload_url,
                        collection_name=upload_collection_name
                    )

                    if response.get('error'):
                        st.error(f"An error occurred: {response['error']}")
                    else:
                        st.success('Data uploaded successfully!', icon='✅')
                        st.subheader('Response:')
                        st.json(response)
            else:
                st.warning('Please enter both a URL and collection name to continue.')

if __name__ == '__main__':
    main()
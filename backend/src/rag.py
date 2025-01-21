import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter

from qdrant import vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
ENV_PATH = '/Users/nikolaynechay/Data_Science/RAG/backend/.env'
if not os.path.exists(ENV_PATH):
    logger.error(f"Environment file not found at {ENV_PATH}")
else:
    load_dotenv(dotenv_path=ENV_PATH)
    logger.info("Environment variables loaded successfully.")

# Retrieve environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_ENDPOINT = os.getenv('QDRANT_ENDPOINT')

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set. Please check your .env file.")
if not QDRANT_API_KEY:
    logger.warning("QDRANT_API_KEY is not set. Some functionalities may not work.")
if not QDRANT_ENDPOINT:
    logger.warning("QDRANT_ENDPOINT is not set. Vector store connection may fail.")

# Initialize the LLM
try:
    LLM = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model='gpt-4o',
        temperature=0
    )
    logger.info("ChatOpenAI model initialized.")
except Exception as e:
    logger.error(f"Failed to initialize ChatOpenAI model: {e}")
    raise

def load_prompt_template(template_path: str, question: str) -> str:
    """
    Load a prompt template from a file and format it with the given context and question.

    Args:
        template_path (str): Path to the template file.
        question (str): Question to be included in the prompt.

    Returns:
        str: Formatted prompt.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found at {template_path}")
    
    try:
        with open(template_path, 'r') as file:
            template = file.read()
        
        formatted_prompt = template.format(question=question)
        logger.info("Prompt template loaded and formatted successfully.")
        return formatted_prompt
    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        raise

def create_chain(template_path: str, question: str):
    """
    Create a chat chain from a template file.

    Args:
        template_path (str): Path to the template file.
        context (str): Context to be included in the chain.
        question (str): Question to be included in the chain.

    Returns:
        Callable: A chain for processing the input.
    """
    try:
        template_prompt = load_prompt_template(template_path, question)
        formatted_prompt = ChatPromptTemplate.from_template(template_prompt)

        logger.info("Initial prompt created successfully.")

        retriever = vector_store.as_retriever()
        retriever_with_config = retriever.with_config(top_k=4)

        chain_response = RunnableParallel(
            {
                'response': formatted_prompt | LLM,
                'context': itemgetter('context')
            }
        )

        chain = {
            'context': retriever_with_config,
            'question': RunnablePassthrough()
        } | chain_response

        logger.info("Chat chain created successfully.")
        return chain
    except Exception as e:
        logger.error(f"Failed to create chat chain: {e}")
        raise

def get_answer(question: str, template_path: str) -> dict:
    """
    Get an answer to a question using the specified template and context.

    Args:
        question (str): The question to be answered.
        template_path (str): Path to the prompt template.
        context (str): Additional context for the question.

    Returns:
        dict: A dictionary containing the answer and context.
    """
    try:
        chain = create_chain(template_path, question)
        response = chain.invoke(question)

        # answer = response['response'].content
        answer = response['response'].content
        retrieved_context = response['context']

        logger.info("Answer retrieved successfully.")
        return {'answer': answer, 'context': retrieved_context}
    except Exception as e:
        logger.error(f"Failed to retrieve answer: {e}")
        raise

if __name__ == "__main__":
    try:
        response = get_answer(
            question="Which 9 states that tax Social Security benefits in 2025 in USA?",
            template_path="/Users/nikolaynechay/Data_Science/RAG/backend/src/prompt_template.txt",
        )
        print(response)
    except Exception as e:
        logger.error(f"Error during main execution: {e}")

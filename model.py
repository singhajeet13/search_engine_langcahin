from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq


load_dotenv()

def create_HF_embedding_model(HF_token: str | None = None):
    """Create HuggingFace embeddings clients."""
    if HF_token is None:
        HF_token = os.getenv("HF_TOKEN")
        

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu", "token": HF_token}
    )
    return embeddings


def create_groq_model(groq_api_key: str | None = None):
    """Create Groq clients."""
    if groq_api_key is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
    
    
    model = ChatGroq(
        model_name="Llama3-8b-8192",
        api_key=groq_api_key,
    )

    return model
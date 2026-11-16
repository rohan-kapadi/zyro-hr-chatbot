"""
Configuration and settings management for Zyro Dynamics HR Help Desk RAG Application.
Supports local development (.env) and Streamlit Community Cloud (st.secrets).
Never exposes API keys in frontend, logs, or error messages.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Load local .env if present (in project root or parent workspace)
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """
    Safely retrieve a secret from Streamlit secrets (if running in Streamlit)
    or from environment variables (.env).
    """
    # Try streamlit secrets first if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    # Fallback to environment variables
    return os.getenv(key, default).strip()


# API Keys and Tracing Configuration
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = get_secret("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
LANGCHAIN_PROJECT = get_secret("LANGCHAIN_PROJECT", "zyro-rag-chatbot")

# Configure LangSmith if enabled and API key is present
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = get_secret("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Document directory resolution
def resolve_policies_dir() -> Path:
    """Find the HR_Policies folder in local project or parent workspace."""
    env_dir = os.getenv("HR_POLICIES_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    candidates = [
        BASE_DIR / "HR_Policies",
        BASE_DIR / "hr_policies",
        BASE_DIR.parent / "HR_Policies",
        BASE_DIR.parent / "hr_policies",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Default fallback
    return BASE_DIR / "HR_Policies"


HR_POLICIES_DIR = resolve_policies_dir()
VECTORSTORE_DIR = BASE_DIR / "faiss_index"

# RAG Pipeline Configuration (Preserving Original Notebook Settings)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVER_SEARCH_TYPE = "mmr"
RETRIEVER_K = 6
RETRIEVER_FETCH_K = 25
RETRIEVER_LAMBDA_MULT = 0.6

# LLM Configuration
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800

# Out-of-Scope Refusal Message
REFUSAL_MESSAGE = (
    "I do not have access to that information in the Zyro Dynamics HR policy documents. "
    "I can only assist with internal company HR policies."
)


def validate_api_keys() -> dict:
    """Return presence status of required keys without revealing actual values."""
    return {
        "google_api_key_configured": bool(get_secret("GOOGLE_API_KEY")),
        "langchain_api_key_configured": bool(get_secret("LANGCHAIN_API_KEY")),
        "langsmith_tracing_enabled": bool(
            get_secret("LANGCHAIN_API_KEY") and
            get_secret("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
        ),
    }

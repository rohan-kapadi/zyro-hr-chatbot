"""
Zyro Dynamics HR Help Desk - RAG Pipeline
Modular implementation of the existing Kaggle notebook RAG architecture.
Preserves chunking, embeddings, FAISS MMR retriever, Gemini LLM, and LCEL chain.
Includes robust citation extraction, 1-based page numbering, and prompt guardrails.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

try:
    from langsmith import traceable
except ImportError:
    # Fallback no-op decorator if langsmith is not installed
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from config import (
    HR_POLICIES_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    EMBEDDING_MODEL_NAME,
    RETRIEVER_SEARCH_TYPE,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    REFUSAL_MESSAGE,
    get_secret,
)

# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------
def load_documents(policies_dir: Optional[Path] = None) -> List[Document]:
    """
    Load all PDF documents from the HR policies directory using PyPDFDirectoryLoader.
    Preserves document metadata (filename, page number, etc.).
    """
    target_dir = Path(policies_dir) if policies_dir else HR_POLICIES_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"HR policies directory not found at: {target_dir}")

    loader = PyPDFDirectoryLoader(str(target_dir))
    documents = loader.load()
    return documents


def list_available_policies(policies_dir: Optional[Path] = None) -> List[str]:
    """Return a sorted list of all PDF filenames in the policies directory."""
    target_dir = Path(policies_dir) if policies_dir else HR_POLICIES_DIR
    if not target_dir.exists():
        return []
    return sorted([f.name for f in target_dir.glob("*.pdf")])


# ---------------------------------------------------------------------------
# 2. Text Chunking
# ---------------------------------------------------------------------------
def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Split loaded documents into chunks preserving notebook parameters (700/120).
    Ensures source and page metadata are retained in each chunk.
    """
    if separators is None:
        separators = CHUNK_SEPARATORS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ---------------------------------------------------------------------------
# 3. Vector Database & Embedding Model
# ---------------------------------------------------------------------------
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Initialize and return the Hugging Face sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vectorstore(
    chunks: Optional[List[Document]] = None,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    persist_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Create or load the FAISS vector database.
    If a local cached index exists and force_rebuild is False, it loads from disk.
    Otherwise, it embeds chunks, creates the index, and saves it locally.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    target_dir = Path(persist_dir) if persist_dir else VECTORSTORE_DIR

    # Load from disk if available and not rebuilding
    index_file = target_dir / "index.faiss"
    if not force_rebuild and index_file.exists():
        try:
            vectorstore = FAISS.load_local(
                folder_path=str(target_dir),
                embeddings=embedding_model,
                allow_dangerous_deserialization=True,
            )
            return vectorstore
        except Exception as e:
            # Fall back to rebuilding if loading fails
            print(f"Failed to load cached FAISS index: {e}. Rebuilding index...")

    # Build from documents
    if chunks is None:
        docs = load_documents()
        chunks = split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Persist locally for fast Streamlit restarts
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(target_dir))
    except Exception as e:
        print(f"Warning: Could not save FAISS index locally: {e}")

    return vectorstore


# ---------------------------------------------------------------------------
# 4. MMR Retriever
# ---------------------------------------------------------------------------
def create_retriever(
    vectorstore: FAISS,
    search_type: str = RETRIEVER_SEARCH_TYPE,
    k: int = RETRIEVER_K,
    fetch_k: int = RETRIEVER_FETCH_K,
    lambda_mult: float = RETRIEVER_LAMBDA_MULT,
):
    """
    Initialize the retriever with Maximum Marginal Relevance (MMR).
    Preserves notebook parameters: k=6, fetch_k=25, lambda_mult=0.6.
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


# ---------------------------------------------------------------------------
# 5. LLM Initialization
# ---------------------------------------------------------------------------
def initialize_llm(
    google_api_key: Optional[str] = None,
    model_name: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> ChatGoogleGenerativeAI:
    """
    Initialize the Gemini LLM with Google GenAI.
    Preserves notebook model and parameters.
    """
    api_key = google_api_key or get_secret("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Please configure it in .env or st.secrets.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# 6. RAG Prompt and LCEL Chain
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """You are a helpful and professional HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the provided policy context below.

Guidelines:
1. Provide a direct, concise, and accurate answer based strictly on the context. If the question asks about eligibility without specifying a grade or role, explain the eligibility rules for the different grades and roles mentioned in the policy.
2. If the question cannot be answered from the HR policies (such as company financial figures/revenue, EXTERNAL products/companies like Zoho or Salesforce, EXTERNAL job applications for other companies, or personal confidential records like individual salary slips), politely refuse to answer.
3. When refusing, clearly state: "I do not have access to that information in the Zyro Dynamics HR policy documents. I can only assist with internal company HR policies."

Context:
{context}

Question:
{question}

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single text context block."""
    return "\n\n".join([doc.page_content for doc in docs])


def create_rag_chain(llm: ChatGoogleGenerativeAI):
    """Create the LCEL RAG execution chain."""
    return RAG_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 7. Citations & Metadata Formatting with Relevance Verification
# ---------------------------------------------------------------------------
STOPWORDS = {
    "that", "this", "with", "from", "have", "were", "been", "their", "they",
    "what", "when", "where", "which", "will", "would", "could", "should",
    "about", "based", "provided", "policy", "documents", "zyro", "dynamics",
    "only", "also", "then", "into", "more", "most", "some", "such", "than",
    "must", "under", "these", "those", "each", "other", "after", "before",
}


def extract_keywords(text: str) -> set:
    """Extract lowercased significant keywords (4+ chars, non-stopword)."""
    words = set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()))
    return words - STOPWORDS


def format_citations(
    docs: List[Document],
    answer: str = "",
    question: str = "",
    is_refusal: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract reliable citation metadata from retrieved Document chunks.
    - Converts 0-based loader page numbers to 1-based page numbers.
    - Cleans and extracts concise text excerpts.
    - Validates supporting relevance so that unrelated retrieved chunks are not
      displayed as confirmed evidence.
    - Returns empty citations list on polite refusal.
    """
    if is_refusal:
        return []

    citations = []
    seen = set()

    answer_keywords = extract_keywords(answer)
    question_keywords = extract_keywords(question)

    for doc in docs:
        source_path = doc.metadata.get("source", "Unknown Document")
        filename = os.path.basename(source_path)

        # Loader uses zero-based page indexing; convert to 1-based page number
        raw_page = doc.metadata.get("page", 0)
        page_num = raw_page + 1 if isinstance(raw_page, int) else 1

        # Use page_label if explicitly provided as a string number
        if "page_label" in doc.metadata:
            try:
                page_num = int(doc.metadata["page_label"])
            except (ValueError, TypeError):
                pass

        content = doc.page_content.strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        clean_text = " ".join(lines)
        excerpt = clean_text[:280] + ("..." if len(clean_text) > 280 else "")

        # Relevance scoring against generated answer
        chunk_words = extract_keywords(clean_text)
        overlap_count = len(chunk_words & answer_keywords)

        # A document chunk is verified as supporting evidence if it shares
        # meaningful content words with the generated answer
        is_supporting = overlap_count >= 2 or (len(docs) == 1)

        key = (filename, page_num, excerpt[:60])
        if key not in seen and is_supporting:
            seen.add(key)
            citations.append({
                "filename": filename,
                "page": page_num,
                "excerpt": excerpt,
                "full_text": content,
                "overlap_score": overlap_count,
            })

    # Sort citations by relevance overlap score descending
    citations.sort(key=lambda x: x.get("overlap_score", 0), reverse=True)

    # Fallback: if strictly no chunk met the threshold but answer is not a refusal,
    # include the top retrieved chunk as best match
    if not citations and docs and not is_refusal:
        top_doc = docs[0]
        fname = os.path.basename(top_doc.metadata.get("source", "Policy Document"))
        raw_p = top_doc.metadata.get("page", 0)
        p_num = raw_p + 1 if isinstance(raw_p, int) else 1
        c_lines = [l.strip() for l in top_doc.page_content.splitlines() if l.strip()]
        citations.append({
            "filename": fname,
            "page": p_num,
            "excerpt": " ".join(c_lines)[:280] + "...",
            "full_text": top_doc.page_content,
            "overlap_score": 1,
        })

    return citations


# ---------------------------------------------------------------------------
# 8. Main Pipeline Query Function
# ---------------------------------------------------------------------------
@traceable(name="ask_hr_helpdesk")
def ask_question(
    question: str,
    retriever,
    chain,
) -> Dict[str, Any]:
    """
    Main function to ask a question to the HR chatbot.
    Retrieves policy context, invokes the RAG chain, and returns structured result.
    """
    # 1. Retrieve relevant policy chunks
    docs = retriever.invoke(question)

    # 2. Format context for prompt
    context = format_docs(docs)

    # 3. Generate answer via LCEL chain
    response_raw = chain.invoke({"context": context, "question": question})
    answer = str(response_raw).strip()

    # 4. Check if answer is an out-of-scope refusal
    refusal_signature = "I do not have access to that information in the Zyro Dynamics HR policy documents"
    is_refusal = refusal_signature.lower() in answer.lower()

    # 5. Format verified citations
    citations = format_citations(docs, answer=answer, question=question, is_refusal=is_refusal)

    return {
        "question": question,
        "answer": answer,
        "sources": citations,
        "retrieved_docs": docs,
        "is_refusal": is_refusal,
    }

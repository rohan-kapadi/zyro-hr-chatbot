"""
Zyro Dynamics | HR Help Desk Chatbot
Production-grade Streamlit application for internal HR policy question answering.
Powered by LangChain, FAISS MMR Retrieval, Hugging Face Embeddings, and Google Gemini.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import rag_pipeline

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zyro Dynamics | HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS styling
CUSTOM_CSS = """
<style>
    /* Global layout & typography */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }
    
    /* Header styling */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .brand-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 9999px;
        border: 1px solid #DBEAFE;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Welcome card */
    .welcome-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .welcome-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .welcome-text {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    
    /* Source Citation Cards */
    .citation-tag {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .source-excerpt {
        background-color: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 8px 12px;
        font-size: 0.86rem;
        color: #334155;
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.5;
        border-radius: 0 6px 6px 0;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    /* Sidebar info box */
    .info-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 14px;
        font-size: 0.82rem;
        color: #475569;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached RAG Pipeline Initialization
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Zyro Dynamics RAG Pipeline...")
def initialize_rag_system(force_rebuild: bool = False):
    """
    Initializes and caches the heavy RAG components:
    embeddings, FAISS index, MMR retriever, Gemini LLM, and LCEL chain.
    """
    # 1. Validate API keys
    api_status = config.validate_api_keys()
    if not api_status["google_api_key_configured"]:
        return None, None, None, "Google API Key is not configured. Please set GOOGLE_API_KEY in .env or st.secrets."

    # 2. Check policy documents
    policies_dir = config.HR_POLICIES_DIR
    if not policies_dir.exists():
        return None, None, None, f"HR Policies folder not found at: {policies_dir}"

    policy_files = list(policies_dir.glob("*.pdf"))
    if not policy_files:
        return None, None, None, f"No PDF documents found in: {policies_dir}"

    # 3. Embedding model and Vector store
    emb_model = rag_pipeline.get_embedding_model()
    vectorstore = rag_pipeline.create_vectorstore(
        embedding_model=emb_model,
        force_rebuild=force_rebuild
    )

    # 4. MMR Retriever
    retriever = rag_pipeline.create_retriever(vectorstore)

    # 5. LLM and LCEL chain
    llm = rag_pipeline.initialize_llm()
    chain = rag_pipeline.create_rag_chain(llm)

    return retriever, chain, len(policy_files), None


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def clear_chat():
    """Clear conversation history."""
    st.session_state.messages = []
    st.session_state.pending_query = None


def submit_suggested_question(q_text: str):
    """Set question from suggested buttons to be processed."""
    st.session_state.pending_query = q_text


# Initialize backend
retriever, chain, policy_count, init_error = initialize_rag_system()


# ---------------------------------------------------------------------------
# Sidebar Interface
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Zyro Dynamics")
    st.markdown("**HR Help Desk Chatbot**")
    st.caption("AI-powered employee assistant for company policies, benefits, and compliance.")

    st.divider()

    # System status
    if init_error:
        st.error("⚠️ Backend Initialization Error")
        st.caption(init_error)
    else:
        st.success(f"🟢 **System Ready** ({policy_count} Policies Loaded)")

    # Available Documents Section
    with st.expander("📁 Available HR Policies", expanded=False):
        available_docs = rag_pipeline.list_available_policies()
        if available_docs:
            for doc_name in available_docs:
                clean_name = doc_name.replace(".pdf", "").replace("_", " ")
                st.markdown(f"- 📄 `{doc_name}`")
        else:
            st.caption("No policy documents found.")

    # Architecture Overview
    with st.expander("⚙️ RAG Architecture Details", expanded=False):
        st.markdown(
            f"""
            - **Loader**: `PyPDFDirectoryLoader`
            - **Chunking**: `RecursiveCharacterTextSplitter` (700 chars, 120 overlap)
            - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-d)
            - **Vector DB**: `FAISS` with local caching
            - **Search**: `MMR` (k={config.RETRIEVER_K}, fetch_k={config.RETRIEVER_FETCH_K}, λ={config.RETRIEVER_LAMBDA_MULT})
            - **LLM**: `{config.LLM_MODEL}` (Temp: {config.LLM_TEMPERATURE})
            - **Chain**: LangChain LCEL + Prompt Guardrails
            """
        )
        keys_status = config.validate_api_keys()
        if keys_status["langsmith_tracing_enabled"]:
            st.info(f"🔍 LangSmith Tracing Active (`{config.LANGCHAIN_PROJECT}`)")

    st.divider()

    # Action buttons
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ Clear Chat", use_container_width=True, on_click=clear_chat):
            pass
    with col_c2:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    st.caption("Confidential — For Internal Use Only • Zyro Dynamics Pvt. Ltd.")


# ---------------------------------------------------------------------------
# Main Chat Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-header">
        <h1 class="brand-title">Zyro Dynamics | HR Help Desk</h1>
        <span class="brand-badge">RAG Assistant</span>
    </div>
    <div class="brand-subtitle">Your intelligent assistant for company HR policies.</div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Welcome Screen (Shown when chat history is empty)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-title">👋 Hello! Welcome to Zyro Dynamics HR Help Desk.</div>
            <div class="welcome-text">
                I can help you understand company policies related to leave entitlements, work-from-home guidelines,
                business travel and expense reimbursements, compensation, code of conduct, and more.<br><br>
                What would you like to know today?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 💡 Suggested Questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌴 What is the annual leave policy?", use_container_width=True):
            submit_suggested_question("What is the annual leave policy?")
            st.rerun()
        if st.button("✈️ How do I claim travel expenses?", use_container_width=True):
            submit_suggested_question("How do I claim travel expenses?")
            st.rerun()
    with col2:
        if st.button("🏠 What is the work-from-home policy?", use_container_width=True):
            submit_suggested_question("What is the work-from-home policy?")
            st.rerun()
        if st.button("📜 What is the code of conduct?", use_container_width=True):
            submit_suggested_question("What is the company code of conduct?")
            st.rerun()


# ---------------------------------------------------------------------------
# Render Existing Conversation History
# ---------------------------------------------------------------------------
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display verified sources & references for assistant messages
        sources = message.get("sources", [])
        if message["role"] == "assistant" and sources:
            with st.expander(f"📚 Sources & References ({len(sources)} Verified Document{'s' if len(sources) > 1 else ''})", expanded=False):
                for src_idx, src in enumerate(sources):
                    st.markdown(
                        f"""
                        **Source {src_idx + 1}:** 📄 `{src['filename']}` &nbsp;|&nbsp; **Page:** `{src['page']}`
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""<div class="source-excerpt">"{src['excerpt']}"</div>""",
                        unsafe_allow_html=True,
                    )

                    # Provide download button for original PDF if accessible
                    pdf_file = config.HR_POLICIES_DIR / src["filename"]
                    if pdf_file.exists():
                        try:
                            with open(pdf_file, "rb") as f:
                                st.download_button(
                                    label=f"📥 Download {src['filename']}",
                                    data=f.read(),
                                    file_name=src["filename"],
                                    mime="application/pdf",
                                    key=f"dl_{msg_idx}_{src_idx}",
                                )
                        except Exception:
                            pass
                    st.markdown("---")


# ---------------------------------------------------------------------------
# Input Handling & Query Processing
# ---------------------------------------------------------------------------
# Check if a suggested question button was clicked or text entered in chat_input
user_query = st.chat_input("Ask a question about Zyro Dynamics HR policies...")
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    if init_error:
        st.error(f"Cannot process request: {init_error}")
    else:
        # 1. Append user message to state and display
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Assistant response generation
        with st.chat_message("assistant"):
            with st.spinner("Searching HR policy documents and generating answer..."):
                try:
                    result = rag_pipeline.ask_question(
                        question=user_query,
                        retriever=retriever,
                        chain=chain,
                    )
                    answer_text = result["answer"]
                    sources = result.get("sources", [])
                    is_refusal = result.get("is_refusal", False)

                    # Display generated answer
                    st.markdown(answer_text)

                    # Display source citations if available and not a refusal
                    if sources and not is_refusal:
                        with st.expander(f"📚 Sources & References ({len(sources)} Verified Document{'s' if len(sources) > 1 else ''})", expanded=True):
                            msg_idx = len(st.session_state.messages)
                            for src_idx, src in enumerate(sources):
                                st.markdown(
                                    f"""
                                    **Source {src_idx + 1}:** 📄 `{src['filename']}` &nbsp;|&nbsp; **Page:** `{src['page']}`
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"""<div class="source-excerpt">"{src['excerpt']}"</div>""",
                                    unsafe_allow_html=True,
                                )
                                pdf_file = config.HR_POLICIES_DIR / src["filename"]
                                if pdf_file.exists():
                                    try:
                                        with open(pdf_file, "rb") as f:
                                            st.download_button(
                                                label=f"📥 Download {src['filename']}",
                                                data=f.read(),
                                                file_name=src["filename"],
                                                mime="application/pdf",
                                                key=f"dl_live_{msg_idx}_{src_idx}",
                                            )
                                    except Exception:
                                        pass
                                st.markdown("---")

                    # Append to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "is_refusal": is_refusal,
                    })

                except Exception as e:
                    error_msg = f"An error occurred while generating the answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "is_refusal": False,
                    })

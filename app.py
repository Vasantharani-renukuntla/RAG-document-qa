"""
AI Document Assistant - Streamlit Web Application
A modern, responsive user interface for document-based question answering using RAG,
FAISS vector search, and Google Gemini API.
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()

from src.rag_pipeline import RAGPipeline, RAGResponse
from src.document_loader import DocumentProcessingError, EmptyDocumentError, FileTooLargeError, UnsupportedFormatError

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Assistant | RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Custom CSS Styling ---
CUSTOM_CSS = """
<style>
    /* Main Theme Overrides & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card Styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid #334155;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Card Container */
    .custom-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.82rem;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 0.5rem;
    }
    .badge-success { background-color: #059669; color: #ecfdf5; }
    .badge-warning { background-color: #d97706; color: #fffbeb; }
    .badge-info { background-color: #2563eb; color: #eff6ff; }
    .badge-score { background-color: #475569; color: #f8fafc; font-family: monospace; }
    
    /* Source Passage Card */
    .source-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 1rem 1.2rem;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 0.8rem;
    }
    .source-title {
        font-weight: 600;
        color: #38bdf8;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    
    /* Hide default streamlit menu items if needed */
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Session State Initialization ---
def init_session_state():
    if "rag_pipeline" not in st.session_state:
        # Load API key from env if available
        default_api_key = os.getenv("GEMINI_API_KEY", "")
        st.session_state.rag_pipeline = RAGPipeline(
            chunk_size=800,
            chunk_overlap=120,
            embedding_model_name="all-MiniLM-L6-v2",
            llm_model_name="gemini-1.5-flash",
            api_key=default_api_key,
            min_similarity_threshold=0.10
        )

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}

    if "last_response" not in st.session_state:
        st.session_state.last_response = None


init_session_state()
pipeline: RAGPipeline = st.session_state.rag_pipeline


# --- Sidebar Setup ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=64)
    st.title("Settings & Upload")
    
    # 1. API Key Input
    st.subheader("🔑 Gemini API Setup")
    env_key = os.getenv("GEMINI_API_KEY", "")
    user_api_key = st.text_input(
        "Enter Gemini API Key",
        value=pipeline.llm_service.api_key or env_key,
        type="password",
        help="Get a free key from Google AI Studio (https://aistudio.google.com/)"
    )
    if user_api_key:
        pipeline.set_api_key(user_api_key)

    st.markdown("---")

    # 2. RAG Hyperparameters Expander
    with st.expander("⚙️ RAG Configurations", expanded=False):
        chunk_size = st.slider("Chunk Size (chars)", min_value=300, max_value=2000, value=pipeline.chunk_size, step=50)
        chunk_overlap = st.slider("Chunk Overlap (chars)", min_value=0, max_value=400, value=pipeline.chunk_overlap, step=10)
        top_k = st.slider("Top K Passages", min_value=1, max_value=10, value=5)
        model_option = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"],
            index=0
        )
        pipeline.llm_service.model_name = model_option
        
        if chunk_size != pipeline.chunk_size or chunk_overlap != pipeline.chunk_overlap:
            pipeline.update_chunking_config(chunk_size, chunk_overlap)

    st.markdown("---")

    # 3. Document Uploader
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Maximum file size limit: 25MB per document."
    )

    if st.button("⚡ Process Uploaded Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one PDF or TXT file to process.")
        else:
            with st.spinner("Processing & indexing documents..."):
                success_count = 0
                for file_obj in uploaded_files:
                    try:
                        bytes_data = file_obj.read()
                        meta = pipeline.ingest_document(file_obj.name, bytes_data)
                        st.session_state.processed_files[file_obj.name] = meta
                        success_count += 1
                    except (UnsupportedFormatError, FileTooLargeError, EmptyDocumentError, DocumentProcessingError) as e:
                        st.error(f"Error in '{file_obj.name}': {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected error in '{file_obj.name}': {str(e)}")

                if success_count > 0:
                    st.success(f"Successfully processed {success_count} document(s)!")
                    st.rerun()

    st.markdown("---")

    # 4. Ingested Documents List
    st.subheader("📋 Processed Documents")
    if pipeline.ingested_documents:
        for filename, meta in list(pipeline.ingested_documents.items()):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 **{filename}** ({meta.file_type}) • {meta.total_pages} page(s) • {meta.char_count} chars")
            with col2:
                if st.button("🗑️", key=f"del_{filename}", help="Remove document"):
                    pipeline.remove_document(filename)
                    if filename in st.session_state.processed_files:
                        del st.session_state.processed_files[filename]
                    st.rerun()
    else:
        st.info("No documents currently indexed.")

    st.markdown("---")

    # 5. Session Reset Button
    if st.button("🔄 Reset Entire Session", use_container_width=True):
        pipeline.clear_all()
        st.session_state.processed_files = {}
        st.session_state.last_response = None
        st.success("Session reset completed!")
        st.rerun()


# --- Main Application Header ---
st.markdown(
    """
    <div class="main-header">
        <h1>AI Document Assistant</h1>
        <p>Document-Based Question Answering System using RAG & Gemini</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Quick Stats Bar ---
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Total Documents", len(pipeline.ingested_documents))
with col_s2:
    st.metric("Total Indexed Chunks", pipeline.vector_store.total_chunks)
with col_s3:
    st.metric("Vector Store", "FAISS CPU (FlatIP)")
with col_s4:
    st.metric("Embedding Model", "all-MiniLM-L6-v2")


# --- Sample Document Loader Option ---
st.markdown("---")
col_samp1, col_samp2 = st.columns([3, 1])
with col_samp1:
    st.markdown("**No documents handy?** Test instantly with our built-in sample document on RAG architecture:")
with col_samp2:
    if st.button("📌 Load Sample Notes"):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_documents", "sample_notes.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                meta = pipeline.ingest_document("sample_notes.txt", f.read())
                st.session_state.processed_files["sample_notes.txt"] = meta
            st.success("Sample notes ingested!")
            st.rerun()
        else:
            st.error("Sample document file not found.")


# --- Document Selection Filter ---
available_docs = list(pipeline.ingested_documents.keys())
selected_docs = None

if available_docs:
    st.subheader("🔍 Search Scope")
    selected_docs = st.multiselect(
        "Select specific documents to search (Leave empty to search ALL processed documents):",
        options=available_docs,
        default=[]
    )
    if not selected_docs:
        selected_docs = None  # None means search all


# --- Question Input & Execution ---
st.subheader("💬 Ask a Question")
question_input = st.text_input(
    "Enter your question regarding the uploaded documents:",
    placeholder="e.g., What is Retrieval-Augmented Generation and how does FAISS work?",
    key="user_question"
)

col_q1, col_q2 = st.columns([1, 4])
with col_q1:
    ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)

if ask_button or (question_input and st.session_state.get("trigger_search", False)):
    if not question_input.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Generating grounded answer from document context..."):
            start_time = time.time()
            response = pipeline.query(
                question=question_input,
                selected_documents=selected_docs,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            st.session_state.last_response = response
            st.session_state.last_elapsed = elapsed


# --- Answer & Results Display Area ---
response: RAGResponse = st.session_state.last_response

if response:
    st.markdown("---")
    st.subheader("💡 Answer")

    if response.status == "success":
        st.markdown(
            f'<span class="badge badge-success">Grounded Answer</span>'
            f'<span class="badge badge-info">{len(response.source_results)} Sources Retrieved</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"### {response.answer}")

    elif response.status == "insufficient_context":
        st.markdown(
            '<span class="badge badge-warning">Insufficient Context</span>',
            unsafe_allow_html=True
        )
        st.warning(response.answer)

    elif response.status == "no_documents":
        st.info(response.answer)

    elif response.status == "error":
        st.error(response.answer)

    # --- Expandable Source Passages & Citations ---
    if response.source_results:
        with st.expander("📚 Expand to Inspect Retrieved Supporting Passages & Citations", expanded=True):
            st.caption(
                "Note: Similarity scores are computed via Cosine Similarity [-1.0 to 1.0] over L2-normalized embeddings, "
                "representing geometric feature similarity (NOT a probability of correctness)."
            )
            for idx, src in enumerate(response.source_results, 1):
                page_str = f" • Page {src.chunk.page_number}" if src.chunk.page_number else ""
                st.markdown(
                    f"""
                    <div class="source-box">
                        <div class="source-title">
                            Source #{idx}: {src.chunk.filename}{page_str}
                            <span class="badge badge-score">Cosine Similarity: {src.similarity_score:.4f}</span>
                        </div>
                        <p style="color: #cbd5e1; font-size: 0.95rem; white-space: pre-wrap; margin: 0;">{src.chunk.text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

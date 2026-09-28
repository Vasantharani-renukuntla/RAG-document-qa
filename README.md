# AI Document Assistant – Document-Based Question Answering System Using RAG

An intermediate-level, production-grade AI Engineering project built with **Python**, **Streamlit**, **PyMuPDF**, **Sentence Transformers**, **FAISS CPU**, and **Google Gemini API**.

---

## 📌 Project Overview & Problem Statement

### Problem Statement
Standard Large Language Models (LLMs) excel at general language understanding, but they lack access to real-time, private, or domain-specific documents. When queried about private contracts, study notes, or enterprise PDFs, standard LLMs may make educated guesses ("hallucinate") rather than providing factual citations.

### Solution
The **AI Document Assistant** solves this problem by implementing a **Retrieval-Augmented Generation (RAG)** architecture. When a user asks a question:
1. The system ingests and indexes uploaded PDF and TXT documents into dense vector representations.
2. The user's query is converted into a vector embedding and searched against a **FAISS CPU** index to retrieve the top $K$ most relevant document chunks.
3. A structured prompt containing **ONLY** the retrieved reference passages is sent to **Google Gemini**.
4. Gemini generates a grounded answer with precise document and page citations.
5. If no relevant evidence is found, the system refrains from calling the LLM or stating unsupported facts, responding with a clear fallback message.

---

## 🚀 Features

- **Multi-Format Document Ingestion**: Ingests digital `.pdf` (using PyMuPDF) and `.txt` files with automatic whitespace normalization and encoding fallbacks.
- **Boundary-Aware Text Chunking**: Splits documents into configurable, overlapping text chunks (default 800 chars, 120 char overlap) respecting sentence and paragraph boundaries.
- **Local Embedding Generation**: Generates 384-dimensional vector embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` with $L2$ normalization.
- **FAISS CPU Vector Store**: Enables fast similarity search using Inner Product ($IP$) cosine similarity over normalized embeddings.
- **Document-Scoped Retrieval**: Allows users to filter queries to specific documents or search across all uploaded files.
- **Grounded Answer Generation**: Uses system instructions to restrict Gemini answers strictly to provided context and prevent hallucinations.
- **Transparent Citation & Source Inspection**: Displays retrieved passages alongside filenames, page numbers, and cosine similarity scores.
- **Session Isolation & State Management**: Manages uploaded files, state, and index per session in Streamlit without permanent disk persistence.
- **Automated Testing Suite**: Includes unit tests with `pytest` covering text extraction, chunking, vector retrieval, and mocked LLM calls.

---

## 🛠️ Technology Stack & Rationale

| Component | Technology | Why It Was Chosen |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Fast, Python-native interactive web application framework. |
| **PDF Extraction** | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | High performance, lightweight text extraction from PDF pages. |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Compact (384d), fast CPU-bound embedding model running locally. |
| **Vector Index** | [FAISS CPU](https://github.com/facebookresearch/faiss) | Extremely efficient vector similarity search library. |
| **LLM Provider** | [Google Gemini API](https://ai.google.dev/) (`google-genai`) | State-of-the-art grounded answer generation. |
| **Config Management**| `python-dotenv` | Clean handling of environment variables and secrets. |
| **Automated Testing**| `pytest` | Modular, simple unit testing framework. |

---

## 📐 Architecture & Workflow

```
┌─────────────────┐       ┌────────────────────┐       ┌──────────────────┐
│ Upload PDF/TXT  │ ────> │ PyMuPDF / Text Ext │ ────> │ Text Chunker     │
└─────────────────┘       └────────────────────┘       └────────┬─────────┘
                                                                │
                                                                v
┌─────────────────┐       ┌────────────────────┐       ┌──────────────────┐
│   User Query    │       │  FAISS Vector Store│ <──── │ SentenceTransf.  │
└────────┬────────┘       └─────────┬──────────┘       │ (all-MiniLM-L6)  │
         │                          │                  └──────────────────┘
         v                          v
┌──────────────────────────────────────────────┐
│ Grounded Prompt Construction & Safety Guard  │
└──────────────────────┬───────────────────────┘
                       │
                       v
┌──────────────────────────────────────────────┐
│  Google Gemini API (Answer Generation)       │
└──────────────────────┬───────────────────────┘
                       │
                       v
┌──────────────────────────────────────────────┐
│  Streamlit UI (Answer & Source Passages)     │
└──────────────────────────────────────────────┘
```

---

## 📋 Prerequisites & Windows Installation

### Prerequisites
- **Python 3.10 or 3.11** installed on Windows.
- **Git** (optional).
- A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).

---

### Step-by-Step Setup Instructions (Windows PowerShell)

#### 1. Open PowerShell and Navigate to Workspace
```powershell
cd d:\B23AI039
```

#### 2. Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
Copy `.env.example` to create a `.env` file:
```powershell
Copy-Item .env.example .env
```
Open `.env` in your text editor and add your **Gemini API Key**:
```env
GEMINI_API_KEY=AIzaSy...your_actual_api_key_here
```

---

## 💻 How to Run the Application

Start the Streamlit web server:
```powershell
streamlit run app.py
```
Once started, open your browser to the local URL (usually `http://localhost:8501`).

---

## 🧪 How to Run Automated Unit Tests

Run all unit tests with `pytest`:
```powershell
pytest -v
```

The test suite runs offline without requiring a live Gemini API key by mocking LLM calls.

---

## 🎯 Example Workflow & Testing Guide

1. **Start Application**: Launch `streamlit run app.py`.
2. **Load Sample Notes**: Click **📌 Load Sample Notes** in the main interface to ingest built-in study notes.
3. **Ask a Document-Grounded Question**:
   - Question: *"What is Retrieval-Augmented Generation?"*
   - Answer: Gemini generates an answer citing `sample_notes.txt`.
4. **Test Unsupported Question (Hallucination Guard)**:
   - Question: *"What is the capital of France?"*
   - Answer: The system responds: *"No sufficiently relevant supporting passage was found in the selected documents to answer this question."* (LLM call skipped).
5. **Upload Custom Documents**: Use the sidebar to upload your own PDFs or TXTs and inspect the retrieved source passages and similarity scores!

---

## ⚠️ Known Limitations & Future Improvements

- **Scanned PDFs**: PDF text extraction relies on digital text layers. Scanned PDFs without OCR will yield empty text. *Future improvement: Integrate Tesseract OCR or pdf2image.*
- **Complex Tables & Images**: Multi-column PDF layouts or diagrams are extracted as continuous text. *Future improvement: Layout-aware PDF parser (e.g. Unstructured).*
- **Local Persistence**: Vector index is stored in memory per session. *Future improvement: Persistent disk saving using FAISS write_index.*

---

## 📄 License
Distributed under the MIT License.

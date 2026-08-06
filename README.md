# 🚀 MediLeaf AI

## 📖 Project Overview

**MediLeaf AI** is an AI-powered assistant designed to help users understand medical leaflet documents quickly and easily.

Users can upload a medicine leaflet (PDF) and ask natural language questions such as:

- Can I take this before food?  
- What are the side effects?  

The system uses a **Retrieval-Augmented Generation (RAG)** pipeline to generate answers **strictly grounded in the uploaded document**, ensuring high accuracy and minimizing hallucinations.

---

## ✨ Features

- 📄 Upload and process medical leaflet PDFs  
- 🤖 Ask questions using natural language  
- 🎯 Answers strictly grounded in document content (RAG)  
- 📊 Confidence-based retrieval filtering  
- 🧠 Structured JSON responses from the LLM  
- 💡 Fast and interactive Streamlit interface  

---

## 🛠️ Technologies Used

- **Python**  
- **Streamlit** (UI)  
- **LangChain**  
- **Google Gemini (LLM)**  
- **SentenceTransformers (E5 Embeddings)**  
- **ChromaDB** (Vector Database)  
- **PyMuPDF (fitz)** for PDF parsing  

---

## 🔍 Retrieval System

The system implements a **semantic search pipeline** using vector embeddings and ChromaDB.

### Configuration

- **Embedding model:** `intfloat/multilingual-e5-base`  
- **Similarity metric:** Cosine similarity  
- **Chunk size:** 800  
- **Chunk overlap:** 150  
- **Top-K retrieval:** 4  

### Retrieval Flow

1. PDF is parsed into structured chunks  
2. Each chunk is embedded and stored in ChromaDB  
3. At query time:
   - Similarity search retrieves top-k chunks  
   - Relevance scores are converted to confidence (%)  
   - Low-confidence results are filtered (<10%)  
   - Boilerplate text is removed  

### Retrieval Enhancements

- ✅ Confidence scoring (0–100%)  
- ✅ Filtering irrelevant chunks  
- ✅ Boilerplate removal (e.g., "consult your doctor")  
- ✅ Context validation (`has_sufficient_context`)  

---

## 📊 Retrieval Evaluation

The retrieval system was evaluated qualitatively based on:

- Relevance of retrieved chunks  
- Confidence score thresholds  
- Noise reduction from filtering  

### Observations

- High-confidence chunks (>30%) consistently contain useful answers  
- Filtering improves answer precision  
- Boilerplate removal reduces irrelevant context  

The selected configuration provided stable and relevant retrieval results.

---

## 🧠 RAG System

MediLeaf AI uses a **Retrieval-Augmented Generation pipeline**:

1. Retrieve relevant chunks  
2. Build structured prompt with context  
3. Generate answer using Google Gemini  

---

## 🧪 RAG Evaluation

Different prompt strategies were tested to ensure reliable responses.

### Final Prompt Design

The system enforces:

- Strict grounding in retrieved context  
- No use of external knowledge  
- JSON-only structured output  
- Explicit handling of missing information  

### Key Prompt Rules

- Use ONLY the provided context  
- Do NOT guess or hallucinate  
- Return `insufficient_information` if answer is missing  
- Respond in the same language as the user  

### Observations

- Strong grounding significantly reduces hallucinations  
- Structured JSON improves response consistency  
- Explicit rules improve reliability in medical use cases  

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/menna-nageh/MediLeaf-AI.git

cd MediLeaf-AI

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows

# Install dependencies
pip install -r requirements.txt
```
##🚀 Usage

```bash
streamlit run streamlit_app.py
```
**Steps**

-Upload a medicine leaflet (PDF)

-Ask your question

-Get an answer grounded in the document

-----

📸 **Demo**

Workflow:

-🧾 Upload leaflet

-❓ Ask question

-💡 Get AI answer


---
📈 **Results**


✅ Fully functional RAG system

✅ Answers strictly grounded in the source document

✅ Reduced hallucination through prompt constraints

✅ Improved accessibility of medical information


---
⚠️ **Limitations**

Answers depend entirely on leaflet quality

No external medical knowledge is used

Retrieval evaluation is qualitative (not quantitative yet)

---
🔮 **Future Improvements**

📊 Quantitative retrieval evaluation (embedding comparison)

🔄 Prompt A/B testing for RAG

🔍 Hybrid search (BM25 + vector search)

🧠 Document re-ranking

📱 Cloud deployment

---
📄 **License**

This project is shared for educational and portfolio purposes.

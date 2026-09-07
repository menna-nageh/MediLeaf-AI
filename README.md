# 🌿 MediLeaf AI

### Ask the leaflet. Get the evidence.
MediLeaf AI is a document-grounded medical leaflet assistant that helps users understand medicine information directly from an uploaded PDF.

Instead of answering from general medical knowledge, the system follows a simple principle:

> **If the leaflet does not support the answer, MediLeaf does not guess.**
Upload a medicine leaflet, ask a question in natural language, and MediLeaf retrieves the most relevant passages before generating a structured answer with source information and grounding checks.

**Educational project — not a diagnostic, prescribing, or clinical decision-making system.**

---

## ✨ What MediLeaf Can Do
📄 **Read medicine leaflets**
Upload one or more PDF leaflets and extract their text page by page.

🔎 **Search beyond exact keywords**
MediLeaf combines semantic vector retrieval with BM25 keyword search to find relevant content.

⚡ **Hybrid retrieval**
Vector search and BM25 results are combined using Reciprocal Rank Fusion (RRF).

🎯 **Rerank the evidence**
An optional Cross-Encoder reranker refines the retrieved candidates before generation.

🧠 **Grounded generation**
Gemini receives the retrieved leaflet context and is explicitly instructed to use that context only.

📌 **Citation-aware answers**
Generated answers reference retrieved source chunk IDs, which are validated against the actual retrieved context.

🛡️ **Grounding verification**
A lightweight lexical grounding check helps detect obviously unsupported generated answers.

📊 **Confidence visibility**
The interface separates retrieval confidence, grounding quality, and overall answer confidence.

💬 **Conversation memory**
A short session-based memory window supports follow-up questions without treating previous conversation as medical evidence.

🚨 **Emergency keyword detection**
Potentially urgent questions can trigger a visible emergency warning while the normal RAG pipeline remains grounded in the leaflet.

👍 **User feedback**
Users can mark answers as helpful or not helpful.

---

# 🧩 How It Works
MediLeaf follows a multi-stage RAG pipeline:

```text
                Medicine PDF
                     |
                     v
                PDF Extraction
                  PyMuPDF
                     |
                     v
                Section-aware
                   Chunking
                     |
               +-----+-----+
               v           v
          Vector Search   BM25
           ChromaDB       Keyword
               +-----+-----+
                     v
                 RRF Hybrid
                     |
                     v
              Cross-Encoder
                Reranking
                     |
                     v
              Grounded Prompt
                     |
                     v
                   Gemini
                     |
               +-----+-----+
               v           v
         Citation Check  Grounding Check
               +-----+-----+
                     v
              Structured Answer
```

---

# 🔍 Retrieval
MediLeaf currently supports four retrieval modes:

| Mode | Description |
|---|---|
| `vector` | Semantic search using ChromaDB |
| `bm25` | Keyword-based retrieval |
| `hybrid` | Vector + BM25 using weighted RRF |
| `hybrid_rerank` | Hybrid retrieval followed by Cross-Encoder reranking |

The default configuration uses:

```dotenv
RETRIEVAL_MODE=hybrid_rerank
```

The system also keeps retrieval metadata such as:

- source file
- page number
- section
- chunk ID
- retrieval method
- retrieval confidence
- reranker score

---

# 🧠 Grounded Generation
The generation layer is designed around a strict rule:

> **Retrieved leaflet content is the evidence.**

The Gemini prompt explicitly instructs the model to:

- use only the retrieved context
- avoid outside medical knowledge
- never guess missing information
- return `insufficient_information` when evidence is inadequate
- preserve the user's language
- return structured JSON
- provide citation IDs for supported claims

The application then validates those citation IDs against the chunks that were actually retrieved.

---

# 📌 Citation & Grounding
Each retrieved chunk receives a stable `SOURCE_ID`.

Example:

```text
SOURCE_ID: leaflet_page_03_chunk_07
FILE: medicine_leaflet.pdf
PAGE: 3
SECTION: Dosage
```

The model can return:

```json
{
  "citations": [
    "leaflet_page_03_chunk_07"
  ]
}
```

MediLeaf validates that the citation actually belongs to the retrieved context before treating it as trusted provenance.

A lightweight lexical grounding check also compares the generated answer against the retrieved text.

---

# 📊 Confidence
MediLeaf intentionally separates different signals instead of presenting one mysterious score.

### Retrieval confidence
How strong the retrieval signal was.

### Grounding score
How well the generated answer overlaps with the retrieved evidence.

### Overall confidence
A combined application-level signal built from retrieval and grounding quality.

These values are shown directly in the Streamlit interface.

> Confidence values are ranking/quality signals, not clinical probabilities.

---

# 💊 Structured Leaflet Summary
After indexing a leaflet, MediLeaf can generate a structured overview including fields such as:

- Medicine name
- Drug class
- Uses
- Contraindications
- Pregnancy
- Breastfeeding
- Children
- Elderly
- Dosage instructions
- Missed dose
- Overdose
- Storage
- Common side effects
- Serious side effects
- Warnings
- When to contact a doctor

Only information available in the leaflet is intended to populate these fields.

---

# 💬 Conversation Memory
MediLeaf keeps a short session-based conversation window to support questions such as:

```text
User:
What are the side effects?

User:
Can children use it?
```

The previous conversation helps resolve references such as **"it"** or **"the medicine"**.

However:

> **Conversation history is not treated as medical evidence.**

Medical facts must still come from the currently retrieved leaflet context.

---

# 🚨 Safety Layer
MediLeaf includes deterministic emergency keyword detection for questions mentioning situations such as:

- overdose
- severe allergic reaction
- difficulty breathing
- chest pain
- seizure
- loss of consciousness
- poisoning

When detected, the UI surfaces an emergency warning encouraging urgent professional care.

This layer does not generate medical advice itself.

---

# 🖥️ Interface
The application is built with Streamlit and includes:

### Main experience

- PDF upload
- automatic leaflet processing
- medicine snapshot
- quick questions
- conversational Q&A
- confidence indicators
- evidence/source display
- grounding status
- answer explanation
- feedback buttons

### Design
The UI uses a dark medical-inspired visual style with:

- glass-style cards
- green accent palette
- structured evidence panels
- compact system status indicators
- responsive Streamlit columns

---

# ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| LLM | Google Gemini |
| LLM framework | LangChain |
| Embeddings | Hugging Face Sentence Transformers |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Reranking | Sentence Transformers Cross-Encoder |
| PDF parsing | PyMuPDF |
| Structured data | Python dataclasses |
| Configuration | `.env` / environment variables |
| API | FastAPI |
| Monitoring | SQLite + JSONL |
| Testing | Pytest |
| Containers | Docker / Docker Compose |

---

# 🔌 API
MediLeaf also exposes a FastAPI layer around the shared application service.

### Health

```text
GET /health
```

### Ask

```text
POST /ask
```

### Streaming

```text
POST /ask/stream
```

The streaming endpoint currently streams the completed structured result as newline-delimited JSON rather than token-by-token Gemini output.

### Feedback

```text
POST /feedback
```

---

# 📈 Evaluation
The project includes retrieval evaluation utilities for comparing:

```text
vector
bm25
hybrid
hybrid_rerank
```

Current metrics include:

### Hit Rate@5
Measures whether the relevant chunk appears in the top five retrieved results.

### MRR@5
Measures how highly the relevant result is ranked.

MediLeaf also includes Prompt A/B evaluation utilities for comparing different prompt versions using metrics such as:

- grounded rate
- citation rate
- mean answer confidence

The current evaluation relies on manually prepared relevant chunk IDs rather than a large versioned benchmark dataset.

---

# 📝 Monitoring
Runtime telemetry is captured locally.

The monitoring layer records information such as:

- question
- answer
- latency
- retrieval mode
- confidence
- grounding result
- insufficient-information state
- feedback

Artifacts are stored through:

```text
logs/
├── JSONL logs
└── medileaf_monitoring.db
```

---

# 🐳 Docker
The repository includes Docker support for running the application stack.

```text
docker compose up --build
```

Default local services:

```text
Streamlit -> http://localhost:8501
FastAPI   -> http://localhost:8000
```

Secrets should remain in a local `.env` file and should never be committed to Git.

---

# 🚀 Local Setup

### 1. Clone

```powershell
git clone https://github.com/menna-nageh/MediLeaf-AI.git
cd MediLeaf-AI
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy blocks activation, run the environment directly:

```powershell
.\.venv\Scripts\python.exe
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment

Copy `.env.example` to `.env`, then configure:

```dotenv
GOOGLE_API_KEY=your_key_here
LLM_MODEL_NAME=gemini-1.5-flash
RETRIEVAL_MODE=hybrid_rerank
TOP_K=5
VECTOR_TOP_K=10
BM25_TOP_K=10
RERANK_TOP_K=5
RERANKER_ENABLED=true
GROUNDING_CHECK_ENABLED=true
CITATION_ENABLED=true
```

### 5. Run the application

```powershell
streamlit run streamlit_app.py
```

---

# 🧪 Testing
Run the test suite with:

```powershell
python -m pytest -q
```

Compile the application:

```powershell
python -m compileall app utils streamlit_app.py
```

The test suite covers areas including:

- PDF parsing
- memory
- emergency detection
- retrieval
- grounding
- citations
- evaluation utilities
- prompts
- API contracts

External model behaviour is intentionally separated from the offline tests.

---

# 🗂️ Project Structure

```text
MediLeaf-AI/
│
├── app/
│   ├── config.py
│   ├── embeddings.py
│   ├── emergency.py
│   ├── llm.py
│   ├── memory.py
│   ├── parser.py
│   ├── prompt_builder.py
│   ├── retriever.py
│   ├── service.py
│   ├── evaluation.py
│   └── api.py
│
├── assets/
├── data/
│   └── pdfs/
├── logs/
├── styles/
├── tests/
├── utils/
├── vector_db/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── streamlit_app.py
├── README.md
└── .env.example
```

---

# ⚠️ Current Limitations
MediLeaf is intentionally honest about what it does not currently provide.

The project currently does **not** include:

- an automated LLM-as-a-judge evaluation pipeline
- a Grafana monitoring dashboard
- a durable on-disk BM25 index
- a batch/watch-based ingestion pipeline
- versioned benchmark evaluation reports
- production-grade API authentication and rate limiting
- token-by-token Gemini streaming

These are potential future improvements rather than features claimed by the current implementation.

---

# 🔮 Future Direction
Possible next steps include:

```text
Real query dataset
        |
        v
Automated benchmark
        |
        v
LLM-as-a-judge
        |
        v
Prompt / model experiments
        |
        v
Persistent monitoring
        |
        v
Production deployment
```

The current architecture is designed so these capabilities can be added without replacing the existing RAG pipeline.

---

# 🎓 Project Goal
MediLeaf AI was built as an applied RAG project to explore how retrieval, reranking, structured generation, citations, and grounding checks can work together around real document-based questions.

The core idea is intentionally simple:

> **Retrieve the evidence first. Generate second.**

---

# 📄 Disclaimer
MediLeaf AI is an educational software project.

It is not a doctor, pharmacist, diagnostic system, prescribing system, or substitute for professional medical advice.

The application is designed to answer from uploaded leaflet content and cannot independently verify whether the leaflet itself is complete, current, or appropriate for a particular patient.

---

# 📜 License
This project is shared for educational and portfolio purposes.
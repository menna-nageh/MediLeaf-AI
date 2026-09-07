# MediLeaf AI

MediLeaf AI is a document-grounded medical leaflet assistant. Users upload a PDF and ask questions answered only from retrieved leaflet text. It is an educational RAG project, not a diagnostic or prescribing system.

## Architecture

```text
PDF -> PyMuPDF extraction -> section-aware chunks -> E5 embeddings -> Chroma
                                                   \-> BM25
Question -> vector/BM25/hybrid RRF -> optional Cross-Encoder reranker
         -> grounded prompt (v1/v2) -> Gemini JSON -> citation validation
         -> lexical grounding check -> Streamlit or FastAPI response
```

`app/service.py` is the shared orchestration boundary used by the API and evaluation code. The Streamlit application retains its existing workflow and UI. JSONL logs and `logs/medileaf_monitoring.db` capture request telemetry.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`. Model selection is configurable without code changes:

```dotenv
LLM_MODEL_NAME=gemini-1.5-flash
# Examples for model comparison, selected one run at a time:
# LLM_MODEL_NAME=gemini-1.5-pro
RETRIEVAL_MODE=hybrid_rerank
```

Run the UI:

```powershell
streamlit run streamlit_app.py
```

## Retrieval Evaluation

`app.evaluation` provides Hit Rate@5 and MRR@5 for `vector`, `bm25`, `hybrid`, and `hybrid_rerank`. Supply a list of `RetrievalExample` values containing a question and gold `chunk_id` set, then call `evaluate_retrieval(store, examples)`. Prompt A/B evaluation compares `v1` and `v2` using grounded rate, citation rate, and mean answer confidence through `evaluate_prompts(...)`.

## API

Start the API with:

```powershell
uvicorn app.api:app --reload
```

Endpoints:

- `GET /health`
- `POST /ask` with `session_id`, `question`, optional `memory_context` and `retrieval_mode`
- `POST /ask/stream` with the same body; returns newline-delimited JSON
- `POST /feedback` with `session_id`, `question`, and boolean `helpful`

The requested `session_id` must refer to an indexed Chroma collection created by the existing ingestion workflow.

## Docker

Keep secrets in a local `.env` file and run:

```powershell
docker compose up --build
```

The API is at `http://localhost:8000`; Streamlit is at `http://localhost:8501`. Local `data`, `logs`, and `vector_db` directories are mounted as volumes.

## Monitoring

Each request records its question, answer, latency, retrieval mode, confidence, grounding result, and insufficient-information state. Feedback is stored with the question and session ID. Runtime artifacts are ignored by Git.

## Rubric Mapping

- RAG pipeline: parsing, chunking, embeddings, Chroma, BM25, hybrid RRF, reranking
- Generation quality: Gemini, grounded prompts, structured output, citations, grounding checks
- Evaluation: Hit Rate@5, MRR@5, retrieval-mode comparison, prompt A/B comparison
- Engineering: shared service layer, configurable model, FastAPI, SQLite monitoring, Docker
- Quality: unit tests for parser, memory, emergency detection, evaluation, grounding/citations, and API contracts

## Implementation Checklist

| Feature | Status | Notes |
|---|---:|---|
| PDF ingestion | ✅ | Streamlit upload workflow |
| PDF parsing | ✅ | PyMuPDF page extraction and cleaning |
| Chunking | ✅ | Structure-aware chunks with provenance |
| Embeddings | ✅ | Configurable HuggingFace embeddings |
| Vector DB / Chroma | ✅ | Persistent Chroma collections per session |
| Semantic retrieval | ✅ | Vector retrieval with relevance filtering |
| BM25 | ✅ | `rank-bm25` dependency and cached corpus |
| Hybrid retrieval / RRF | ✅ | Weighted reciprocal rank fusion |
| Cross-encoder reranking | ✅ | Optional configurable reranker |
| Citation IDs | ✅ | Validated source chunk IDs |
| Grounding check | ✅ | Lexical heuristic grounding score |
| Structured LLM output | ✅ | Defensive JSON normalization |
| Conversation memory | ✅ | Bounded session memory, excluded from evidence |
| Emergency detection | ✅ | Emergency keyword and phrase detection |
| Streamlit UI | ✅ | PDF upload, chat, summary, feedback |
| Better UI / source display | ✅ | Confidence, sources, citations, retrieval details |
| Quantitative retrieval evaluation | ✅ | Hit Rate@5 and MRR@5 utilities |
| Retrieval comparison experiments | ✅ | Vector, BM25, hybrid, hybrid_rerank |
| RAG evaluation / LLM judge | ⚠️ | Grounding and citation checks exist; LLM judge not yet automated |
| Prompt A/B testing | ✅ | Prompt versions `v1` and `v2` |
| Model comparison | ✅ | `LLM_MODEL_NAME` environment configuration |
| FastAPI API | ✅ | `/ask`, `/feedback`, `/health` |
| Streaming API | ✅ | `/ask/stream` newline-delimited JSON response |
| Monitoring DB | ✅ | SQLite request and feedback records |
| Grafana dashboard | ❌ | Not included; SQLite and JSONL data are ready for a future dashboard |
| Docker Compose | ✅ | API and Streamlit services |
| Automated ingestion pipeline | ⚠️ | Upload-driven ingestion exists; batch watcher not yet included |
| Persistent BM25 index | ⚠️ | Cached per process; durable on-disk BM25 index not yet included |
| Deployment setup | ✅ | Docker Compose and Uvicorn configuration |
| Security / rate limiting / API auth | ⚠️ | API key setting exists but request enforcement and rate limiting remain |
| Reproducible evaluation artifacts | ⚠️ | Evaluation utilities exist; versioned gold datasets/reports remain |
| Large automated test suite | ✅ | Unit and API contract coverage included; external model tests are mocked or omitted |

The checklist deliberately distinguishes implemented features from planned
production hardening. The incomplete items are limitations, not hidden claims
about the current system.

## Limitations

- Evaluation quality depends on a manually prepared gold set of relevant chunk IDs.
- `/ask/stream` streams the completed structured result; Gemini token streaming is not enabled yet.
- The system does not replace a clinician and cannot verify leaflet accuracy, patient-specific safety, or current regulatory guidance.
- Cross-Encoder and embedding models may require significant CPU memory on a small Windows machine.

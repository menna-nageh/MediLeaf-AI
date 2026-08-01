# 🚀 MediLeaf AI

> AI Medication Leaflet Assistant

# 📖 Project Overview

MediLeaf AI is an intelligent assistant for medicine package leaflets.
Users upload a PDF leaflet, then ask natural-language questions such as
"Can I take this before food?" or "What are the common side effects?".

The assistant answers strictly from the uploaded leaflet only, with no
medical diagnosis, no prescription, and no external general knowledge.
Every response is grounded in the document and cites the source page.
The system ensures safe, transparent, and reliable answers for everyday users.

---

# ✨ Features

* Upload one or more medicine leaflet PDFs.
* Automatic text extraction, chunking, and semantic search over the leaflet.
* Answer questions using only leaflet content.
* Source citations and retrieval confidence shown for every answer.
* Automatic leaflet summary after upload.
* Emergency keyword detection to highlight urgent phrases safely.
* Follow-up question memory for short conversational context.
* 👍 / 👎 feedback on answers.

---

# 🛠️ Technologies Used

* Python 3.10+
* Streamlit for the web interface
* LangChain for RAG orchestration
* ChromaDB for vector search
* HuggingFace embeddings (`BAAI/bge-small-en-v1.5`)
* Google Gemini via `langchain-google-genai`
* PyMuPDF for PDF parsing
* `python-dotenv` for environment configuration
* `pytest` for unit testing

---

# ⚙️ Installation

1. Clone or unzip the repository and open the `MediLeaf-AI` folder.
2. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the environment template and add your API key:

```bash
copy .env.example .env
```

5. Open `.env` and set `GOOGLE_API_KEY` to your Gemini API key.

---

# 🚀 Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Then in your browser:

1. Upload one or more medicine leaflet PDFs.
2. Click **Process leaflet(s)**.
3. Read the generated leaflet summary.
4. Ask questions in the chat box or use the quick question buttons.
5. Review the answer, source citation, and confidence score.

---

# 📸 Demo

Open the app in your browser after running Streamlit, upload a leaflet PDF,
and ask questions like:

* "Can I take this before food?"
* "What are the common side effects?"
* "Can children use this medicine?"

The assistant responds using only the uploaded leaflet.

---

# 📈 Results

MediLeaf AI makes medicine leaflets easier to understand by:

* Reducing the time needed to find dosage, warnings, and side-effect details.
* Helping users get clear, leaflet-specific answers in plain language.
* Avoiding hallucination by answering only from the uploaded document.
* Preserving source transparency with page citations.

---

# 🔮 Future Improvements

* Add OCR support for scanned leaflet PDFs.
* Support multiple languages and translated answers.
* Add mobile-friendly UI and voice question support.
* Allow users to compare multiple leaflets side-by-side.

---


# 📄 License

This project is shared for educational and portfolio purposes.

---

## Submission Guidance

Please submit the full `MediLeaf-AI` project folder, including:

- `README.md`
- `requirements.txt`
- `streamlit_app.py`
- `app/`
- `utils/`
- `data/`
- `styles/`
- `assets/`
- `tests/`

### Notes:
- Do NOT include generated files such as `vector_db/`
- Do NOT include uploaded PDF files inside `data/pdfs/`
- Ensure the project runs correctly using the provided `requirements.txt`

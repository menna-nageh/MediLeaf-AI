# 🚀 MediLeaf AI

# 📖 Project Overview

**MediLeaf AI** is an intelligent AI-powered assistant that helps users understand medicine leaflets بسهولة وسرعة.

Users can upload a medicine leaflet (PDF) and ask questions مثل:

* *Can I take this before food?*
* *What are the side effects?*

The system answers **only based on the leaflet content**, ensuring accuracy and reliability.

---

# ✨ Features

* 📄 Upload and process medical leaflet PDFs
* 🤖 Ask questions using natural language
* 🎯 Answers strictly grounded in document content (RAG)
* 🌐 Supports clear and simple explanations
* 💡 Fast and interactive Streamlit interface

---

# 🛠️ Technologies Used

* **Python**
* **Streamlit** (UI)
* **LangChain**
* **Google Gemini (LLM)**
* **SentenceTransformers / BGE Embeddings**
* **ChromaDB** (Vector Database)
* **PyMuPDF (fitz)** for PDF processing

---

# ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/menna-nageh/MediLeaf-AI.git

# Navigate to project folder
cd MediLeaf-AI

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

# 🚀 Usage

```bash
streamlit run streamlit_app.py
```

Then:

1. Upload a medicine leaflet (PDF)
2. Ask your question
3. Get accurate answers instantly

---

# 📸 Demo

🎯 Quick Preview
MediLeaf AI in action — upload, ask, and get answers instantly.

🧾 Step 1: Upload Medicine Leaflet

❓ Step 2: Ask Your Question

💡 Step 3: Get AI Answer

---

# 📈 Results

* Successfully built a **working RAG system** for medical documents
* Accurate answers limited strictly to leaflet content
* Improved accessibility of complex medical information
* Clean and user-friendly interface

---

# 🔮 Future Improvements

* 🌍 Add multilingual support (Arabic/English fully)
* 📱 Deploy as a web/mobile app
* 🧠 Improve answer explanations with summaries
* 📊 Add confidence scoring for answers

---

---

# 📄 License

This project is shared for educational and portfolio purposes.
orrectly using the provided `requirements.txt`

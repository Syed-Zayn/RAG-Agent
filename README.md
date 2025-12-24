# 🧠 Intelligent RAG Assistant (Production-Ready Prototype)

A **secure, explainable, and production-grade Retrieval-Augmented Generation (RAG)** system built for intelligent knowledge retrieval from internal corporate documents.

Unlike basic RAG demos, this project emphasizes **security, transparency, confidence scoring, and hallucination control**, making it suitable for **real-world enterprise use cases**.

---

## 🌐 Live Links

🔹 **Live Demo (Streamlit):** _[Add your Streamlit Cloud link here]_  
🔹 **Backend API:**  

https://rag-agent-production-a165.up.railway.app


🔹 **GitHub Repository:**  


https://github.com/Syed-Zayn/RAG-Agent


---

## ✨ Key Highlights

✅ **Enterprise-Ready RAG Architecture**  
✅ **Role-Based Access Control (RBAC)**  
✅ **Explainable AI with Confidence Scoring**  
✅ **Grounded Answers with Source Citations**  
✅ **Persistent Vector Store & Chat History**

---

## 🚀 Features

### 📄 Document Ingestion
- Upload **PDF** and **TXT** files
- Recursive text chunking for optimal embeddings
- Metadata preserved (file name, page number, owner)

---

### 🔒 Role-Based Privacy (RBAC)
- **Private Documents:** Visible only to the uploader
- **Public Documents:** Shared across all users
- Secure filtering during retrieval

---

### 🎯 Confidence Scoring (Explainability)
- Each response includes a **confidence percentage**
- Calculated using **vector similarity distance**
- Helps users judge reliability of answers

---

### 🔍 Transparent Citations (Anti-Hallucination)
- Every answer includes:
  - Source document name
  - Page number / chunk reference
- Reduces hallucinations and improves trust

---

### 💾 Persistent Storage
- **FAISS Vector Index** persisted on disk
- **SQLite Database** for chat history
- Data stored via **Railway Volumes**

---

## 🧠 Architecture Overview



User (Streamlit UI)
|
v
FastAPI Backend (Async)
|
v
LangChain Orchestration
|
v
FAISS Vector Store (Persistent)
|
v
LLM (OpenAI / Gemini)


---

## 🛠️ Tech Stack

| Layer        | Technology |
|--------------|------------|
| Frontend     | Streamlit (Python) |
| Backend      | FastAPI (Async/Await) |
| Orchestration| LangChain |
| Vector Store | FAISS (Local Disk Persistence) |
| Database     | SQLite |
| LLMs         | OpenAI GPT-3.5-Turbo / Google Gemini 1.5 Flash |
| Deployment   | Railway (Dockerized) |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Syed-Zayn/RAG-Agent.git
cd RAG-Agent

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Environment Variables

Create a .env file:

OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key

5️⃣ Run Backend (FastAPI)
uvicorn main:app --reload

6️⃣ Run Frontend (Streamlit)
streamlit run app.py

🔐 Security Considerations

Document-level access control enforced at retrieval time

No unauthorized cross-user document leakage

Ready to integrate authentication (JWT / OAuth)

📈 Future Enhancements

🔑 User Authentication (JWT)

🗂️ Multi-tenant Organizations

🧠 Hybrid Search (BM25 + Vector)

📊 Admin Dashboard & Analytics

🧾 Audit Logs

🧠 Custom LLM Fine-Tuning

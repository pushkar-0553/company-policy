# 📋 AI Company Policy Assistant (RAG)

An enterprise-ready Retrieval-Augmented Generation (RAG) web application that provides accurate, context-grounded answers to employee company policy questions.

Built with **Flask**, **LangChain**, **FastEmbed**, **Groq LLM (`qwen/qwen3.8-27b`)**, and **Qdrant Vector Database**. Ready for 1-click hosting on **Render**.

---

## 🚀 Features

- **Conversational UI**: Sleek, mobile-responsive chat interface with quick suggestion chips, message bubbles, and loading indicators.
- **Pure Cloud & Lightweight**:
  - **Chat LLM**: Powered by Groq's LPUs for sub-second responses.
  - **Embeddings**: Uses `FastEmbed` (`BAAI/bge-small-en-v1.5`), requiring **zero embedding API keys** and running in lightweight ONNX memory (< 100MB RAM, perfect for Render's 512MB free tier).
- **Qdrant Cloud Integration**: Automatic vector store indexing and cosine similarity retrieval.
- **Production Ready**: Includes `Procfile`, `render.yaml`, `gunicorn`, and clean UTF-8 dependency configurations.

---

## 🛠️ Project Structure

```text
rag-project/
├── app.py                      # Flask web server & API endpoints (/ and /api/chat)
├── rag_engine.py               # Core RAG logic (FastEmbed + Groq + Qdrant)
├── ingest.py                   # Automated indexing & verification script
├── ai_company_policy_assistant.py # Terminal version
├── policy.txt                  # Company policy document
├── templates/
│   └── index.html              # Chat UI template
├── static/
│   ├── css/style.css           # Modern styles & responsive layout
│   └── js/chat.js              # Real-time chat handler
├── requirements.txt            # Python dependencies (UTF-8)
├── Procfile                    # Render web process entrypoint
├── render.yaml                 # Render infrastructure-as-code
├── .env.example                # Example environment variables
└── .gitignore                  # Git ignore rules (.venv, .env, etc.)
```

---

## ⚙️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "rag project"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure `.env`**:
   Copy `.env.example` to `.env` and fill in your credentials:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   QDRANT_CLUSTER_ENDPOINT=https://your-cluster-url.cloud.qdrant.io
   QDRANT_KEY=your_qdrant_api_key_here
   QDRANT_COLLECTION=company_policy
   ```
   > *Get a free Groq API key at: [https://console.groq.com/keys](https://console.groq.com/keys)*  
   > *Create a free Qdrant cloud cluster at: [https://cloud.qdrant.io](https://cloud.qdrant.io)*

5. **Index the policy data (One-time check)**:
   ```bash
   python ingest.py
   ```

6. **Start the Flask web app**:
   ```bash
   python app.py
   ```
   Open **`http://127.0.0.1:5000`** in your browser.

---

## 🌐 Deploying to Render

1. Push your repository to **GitHub**:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```

2. Log into **[Render](https://dashboard.render.com)** and create a new **Web Service**:
   - Select your GitHub repository.
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`

3. Add the following **Environment Variables** in Render's dashboard:
   - `GROQ_API_KEY`
   - `QDRANT_CLUSTER_ENDPOINT`
   - `QDRANT_KEY`
   - `QDRANT_COLLECTION`

4. Click **Deploy Web Service**.

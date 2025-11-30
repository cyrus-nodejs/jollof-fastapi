Title: Embedding Service API
Author: Emmanuel Adeyemi
Version: 1.0.0

# 🚀 Embedding Service API
Fast, lightweight text-embedding & cosine-similarity service built with **FastAPI** and **FastEmbed**.

Optimized for low-RAM environments (e.g., Render Free Tier).

---

# 📌 Features

- ⚡ Ultra-lightweight embedding model: `BAAI/bge-small-en-v1.5`
- 🧠 Generate high-quality text embeddings
- 📐 Compute cosine similarity between vectors
- 🔌 FastAPI routes:
    - `GET /` — Health check
    - `POST /embed` — Generate embedding
    - `POST /similarity` — Compute cosine similarity
- 🛡 Typed Pydantic models
- 🐳 Easy deployment (Render, Docker)

---

# 🏗 Project Structure

```text
├── main.py
├── requirements.txt
└── README.fmd

```


---

## 📦 Installation

### 1. Clone the project

```bash
git clone <your-repo-url>
cd embedding-service

```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install dependencies
```bash
pip install -r requirements.txt

```

### 4. Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
---
title: Pharmaand Backend
emoji: 💊
colorFrom: purple
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Pharmaand RAG Backend

A FastAPI-powered RAG (Retrieval-Augmented Generation) backend for the Pharmaand AI chatbot.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your_key_here
```

4. Extract website content:
```bash
python extract_content.py
```

5. Build vector database:
```bash
python build_db.py
```

6. Run locally:
```bash
uvicorn main:app --reload
```

Visit http://localhost:8000/docs for API documentation.

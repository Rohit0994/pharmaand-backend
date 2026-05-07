from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import ask_question

# Initialize FastAPI app
app = FastAPI(
    title="Pharmaand Backend",
    description="RAG-powered backend for Pharmaand AI chatbot",
    version="1.0.0"
)

# Enable CORS for the chatbot widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list

# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Pharmaand RAG Backend",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health"
        }
    }

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Main Q&A endpoint
@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """
    Ask a question and get an AI-generated answer based on Pharmaand's content.
    """
    try:
        result = ask_question(request.question)
        return QuestionResponse(**result)
    except Exception as e:
        return QuestionResponse(
            answer=f"Error processing your question: {str(e)}",
            sources=[]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

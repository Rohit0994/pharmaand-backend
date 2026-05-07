import chromadb
from google import genai
from dotenv import load_dotenv
import os

# Load environment variables (.env for local, HF Secrets for production)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Don't crash on startup — fail gracefully at request time
    print("⚠️ WARNING: GEMINI_API_KEY not set. Add it as a secret in HF Spaces settings.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Initialize ChromaDB with built-in default embeddings (no Rust/sentence-transformers)
CHROMA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="pharmaand_docs")

def search_documents(query, top_k=3):
    """Search ChromaDB for relevant documents."""
    results = collection.query(query_texts=[query], n_results=top_k)
    
    if not results or not results["documents"][0]:
        return []
    
    documents = []
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i].get("source", "unknown")
        documents.append({"content": doc, "source": source})
    
    return documents

def generate_answer(query, documents):
    """Generate answer using Gemini."""
    if not gemini_client:
        return "Backend configuration error: GEMINI_API_KEY is not set. Please add it in HF Spaces secrets."

    context = "\n\n".join([f"Source: {doc['source']}\n{doc['content']}" for doc in documents])
    
    prompt = f"""You are a helpful assistant for Pharmaand GmbH. 
    
Using the following context from our website and documentation, answer the user's question:

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a clear, helpful answer based on the context. If the context doesn't contain relevant information, say so politely."""
    
    response = gemini_client.models.generate_content(
        model="models/gemini-2.0-flash-lite",
        contents=prompt
    )
    return response.text

def ask_question(question):
    """Complete RAG pipeline: search + answer."""
    print(f"🔍 Searching for relevant documents...")
    documents = search_documents(question, top_k=3)
    
    if not documents:
        return {
            "answer": "I couldn't find relevant information in our database to answer your question. Please try rephrasing or contact our support team.",
            "sources": []
        }
    
    print(f"📄 Found {len(documents)} relevant documents")
    print(f"💭 Generating answer...")
    
    answer = generate_answer(question, documents)
    
    sources = list(set([doc["source"] for doc in documents]))
    
    return {
        "answer": answer,
        "sources": sources
    }

# Test function
if __name__ == "__main__":
    test_question = "What products does Pharmaand offer?"
    result = ask_question(test_question)
    print(f"\n✅ Answer: {result['answer']}")
    print(f"📚 Sources: {result['sources']}")

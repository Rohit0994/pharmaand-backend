import chromadb
import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables (.env for local, HF Secrets for production)
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    print("⚠️ WARNING: NVIDIA_API_KEY not set. Add it as a secret in HF Spaces settings.")

# NVIDIA Mistral API configuration
NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

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
    """Generate answer using NVIDIA Mistral API."""
    if not NVIDIA_API_KEY:
        return "Backend configuration error: NVIDIA_API_KEY is not set. Please add it in HF Spaces secrets."

    context = "\n\n".join([f"Source: {doc['source']}\n{doc['content']}" for doc in documents])

    prompt = f"""You are a helpful assistant for Pharmaand GmbH.

Using the following context from our website and documentation, answer the user's question:

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a clear, helpful answer based on the context. If the context doesn't contain relevant information, say so politely."""

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-small-4-119b-2603",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.10,
        "top_p": 1.00,
        "stream": False
    }

    try:
        response = requests.post(NVIDIA_INVOKE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "Error: Unexpected response format from API"
    
    except requests.exceptions.RequestException as e:
        return f"Error calling NVIDIA API: {str(e)}"

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

import chromadb
import requests
import json
from dotenv import load_dotenv
import os
from chromadb.utils import embedding_functions

# Load environment variables (.env for local, HF Secrets for production)
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    print("⚠️ WARNING: NVIDIA_API_KEY not set. Add it as a secret in HF Spaces settings.")

# NVIDIA Mistral API configuration
NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Initialize ChromaDB with SentenceTransformer embeddings
CHROMA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="pharmaand_docs",
    embedding_function=embed_fn
)

TOP_K = 4  # Retrieve 4 documents for better context

def search_documents(query, top_k=TOP_K):
    """Search ChromaDB for relevant documents."""
    results = collection.query(query_texts=[query], n_results=top_k)
    
    if not results or not results["documents"][0]:
        return []
    
    documents = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        documents.append({
            "content": doc,
            "page": metadata.get("page", "unknown"),
            "title": metadata.get("title", "Unknown Page"),
            "url": metadata.get("url", "#")
        })
    
    return documents

def generate_answer(query, documents):
    """Generate answer using NVIDIA Mistral API with strict constraints."""
    if not NVIDIA_API_KEY:
        return "Backend configuration error: NVIDIA_API_KEY is not set. Please add it in HF Spaces secrets."

    context = "\n\n".join([f"Source: {doc['title']}\n{doc['content']}" for doc in documents])

    prompt = f"""You are a helpful assistant for Pharmaand GmbH.

Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, say so politely and suggest contacting support@pharmaand.com.
Keep answers concise (2-5 sentences). Use plain language.
Never invent products, prices, or medical advice.

CONTEXT:
{context}

USER QUESTION: {query}

Answer:"""

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
    documents = search_documents(question, top_k=TOP_K)
    
    if not documents:
        return {
            "answer": "I couldn't find relevant information in our database to answer your question. Please try rephrasing or contact our support team.",
            "sources": []
        }
    
    print(f"📄 Found {len(documents)} relevant documents")
    print(f"💭 Generating answer...")
    
    answer = generate_answer(question, documents)
    
    # Deduplicate sources by URL
    seen = set()
    sources = []
    for doc in documents:
        url = doc["url"]
        if url not in seen:
            seen.add(url)
            sources.append({
                "title": doc["title"],
                "url": url
            })
    
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

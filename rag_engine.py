import chromadb
import requests
import json
from dotenv import load_dotenv
import os
from chromadb.utils import embedding_functions

# Load environment variables (.env for local, HF Secrets for production)
load_dotenv()

# Azure OpenAI configuration - all values come from HF Space secrets/variables.
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

if not AZURE_OPENAI_API_KEY:
    print("⚠️ WARNING: AZURE_OPENAI_API_KEY not set. Add it as a secret in HF Spaces settings.")
if not AZURE_OPENAI_ENDPOINT:
    print("⚠️ WARNING: AZURE_OPENAI_ENDPOINT not set (e.g. https://<resource>.openai.azure.com).")
if not AZURE_OPENAI_DEPLOYMENT:
    print("⚠️ WARNING: AZURE_OPENAI_DEPLOYMENT not set (the deployment name you gave the model in Azure).")

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
    """Generate answer using Azure OpenAI Chat Completions API with strict constraints."""
    if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT):
        return (
            "Backend configuration error: set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "and AZURE_OPENAI_DEPLOYMENT in HF Spaces secrets/variables."
        )

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
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }

    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.10,
        "top_p": 1.00,
        "stream": False,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code >= 400:
            return (
                f"Error calling Azure OpenAI: HTTP {response.status_code} for deployment "
                f"'{AZURE_OPENAI_DEPLOYMENT}'. Body: {response.text[:500]}"
            )
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "Error: Unexpected response format from API"

    except requests.exceptions.RequestException as e:
        return f"Error calling Azure OpenAI: {str(e)}"

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

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# Parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
CHROMA_PATH = "chroma_db"
DATA_FILE = "content/data.json"

def build_vector_db():
    """Build ChromaDB vector database from extracted content using SentenceTransformer."""

    # Load content
    if not os.path.exists(DATA_FILE):
        print(f"❌ {DATA_FILE} not found. Run extract_content.py first.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content_dict = json.load(f)

    print(f"📚 Loaded content from {len(content_dict)} pages")

    # Initialize ChromaDB with SentenceTransformer embeddings (semantic similarity)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # 384-dim, optimized for NLP tasks
    )

    try:
        client.delete_collection(name="pharmaand_docs")
    except:
        pass

    collection = client.get_or_create_collection(
        name="pharmaand_docs",
        embedding_function=embed_fn
    )

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print(f"\n🔄 Splitting content into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    all_chunks = []
    chunk_id = 0

    for page_name, page_data in content_dict.items():
        # Handle both old (string) and new (dict) formats
        if isinstance(page_data, dict):
            content = page_data["content"]
            title = page_data["title"]
            url = page_data["url"]
        else:
            content = page_data
            title = page_name
            url = f"https://rohit0994.github.io/pharmaand_GmBH/{page_name}.html"
        
        chunks = splitter.split_text(content)
        print(f"  {page_name}: {len(chunks)} chunks")
        for chunk in chunks:
            all_chunks.append({
                "id": str(chunk_id),
                "text": chunk,
                "page": page_name,
                "title": title,
                "url": url
            })
            chunk_id += 1

    print(f"\n✅ Total chunks: {len(all_chunks)}")
    print(f"\n⚙️ Storing chunks in ChromaDB (using SentenceTransformer embeddings)...")

    # Add in batches of 50
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        metadatas = [{"page": item["page"], "title": item["title"], "url": item["url"]} for item in batch]
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
    
    print(f"\n✅ Database created successfully!")
    print(f"   Total chunks stored: {len(all_chunks)}")

if __name__ == "__main__":
    build_vector_db()

if __name__ == "__main__":
    build_vector_db()

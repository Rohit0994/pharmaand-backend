import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
CHROMA_PATH = "chroma_db"
DATA_FILE = "content/data.json"

def build_vector_db():
    """Build ChromaDB vector database from extracted content."""

    # Load content
    if not os.path.exists(DATA_FILE):
        print(f"❌ {DATA_FILE} not found. Run extract_content.py first.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content_dict = json.load(f)

    print(f"📚 Loaded content from {len(content_dict)} pages")

    # Initialize ChromaDB with default embedding function (no Rust required)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name="pharmaand_docs")
    except:
        pass

    collection = client.get_or_create_collection(name="pharmaand_docs")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    print(f"\n🔄 Splitting content into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    all_chunks = []
    chunk_id = 0

    for page_name, content in content_dict.items():
        chunks = splitter.split_text(content)
        print(f"  {page_name}: {len(chunks)} chunks")
        for chunk in chunks:
            all_chunks.append({"id": str(chunk_id), "text": chunk, "source": page_name})
            chunk_id += 1

    print(f"\n✅ Total chunks: {len(all_chunks)}")
    print(f"\n⚙️ Storing chunks in ChromaDB (using built-in embeddings)...")

    # Add in batches of 50
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch]
        )
        print(f"  Stored {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

    print(f"\n✅ Vector database built successfully!")
    print(f"📁 Saved to: {CHROMA_PATH}")

if __name__ == "__main__":
    build_vector_db()

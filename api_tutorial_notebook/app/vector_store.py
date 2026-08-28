import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Define where to save the vector database locally
CHROMA_PERSIST_DIR = "./chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# 2. Initialize the local embedding model
# nomic-embed-text is the 2026 standard for lightweight, high-quality local embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Initialize ChromaDB
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="knowledge_notes"
)

print("✅ Vector Store initialized successfully.")

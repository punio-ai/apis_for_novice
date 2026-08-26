import uuid  # Add this to your imports at the top of main.py
from fastapi import HTTPException
import uuid
from langfuse import Langfuse

import httpx
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func
from pydantic import BaseModel

from app.processing import extract_text_from_pdf, chunk_text
from app.database import init_db, get_db
from app.models import DocumentChunk
from app.config import settings
from app.retrieval import hybrid_search
from app.ollama_client import get_embeddings_batch  # <-- NEW IMPORT


app = FastAPI(title="Production RAG LLMOps API (Ollama)")


@app.on_event("startup")
async def startup():
    await init_db()

# --- ENDPOINTS ---


class IngestRequest(BaseModel):
    document_name: str
    text: str


@app.post("/ingest")
async def ingest_document(payload: IngestRequest, db: AsyncSession = Depends(get_db)):
    # Import get_embedding locally or from ollama_client if still needed
    from app.ollama_client import get_embedding
    embedding_vector = await get_embedding(payload.text)

    new_chunk = DocumentChunk(
        document_name=payload.document_name,
        chunk_text=payload.text,
        embedding=embedding_vector
    )

    db.add(new_chunk)
    await db.commit()
    await db.refresh(new_chunk)

    return {
        "message": "Chunk stored successfully",
        "id": new_chunk.id,
        "dimensions": len(embedding_vector)
    }


@app.post("/ingest-file")
async def ingest_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are currently supported.")

    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text.strip():
        raise HTTPException(
            status_code=400, detail="Could not extract text from PDF. Is it scanned/image-based?")

    chunks = chunk_text(raw_text)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="No chunks generated from text.")

    all_embeddings = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = await get_embeddings_batch(batch_chunks)
        all_embeddings.extend(batch_embeddings)

    db_objects = [
        DocumentChunk(
            document_name=file.filename,
            chunk_text=chunk_text_item,
            embedding=embedding_vector,
            text_search=func.to_tsvector('english', chunk_text_item)
        )
        for chunk_text_item, embedding_vector in zip(chunks, all_embeddings)
    ]

    db.add_all(db_objects)
    await db.commit()

    return {
        "message": "File processed successfully",
        "filename": file.filename,
        "total_chunks": len(chunks),
        "dimensions": len(all_embeddings[0]) if all_embeddings else 0
    }


class QueryRequest(BaseModel):
    query: str


async def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a precise AI assistant. Answer the user's question using ONLY the provided context. 
If the answer is not explicitly stated in the context, reply with: "I do not have enough information in the provided context to answer that."

Context:
{context}

Question: {query}
"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"LLM Generation failed: {str(e)}")


@app.post("/query")
async def query_system(payload: QueryRequest, db: AsyncSession = Depends(get_db)):
    trace_id = str(uuid.uuid4())  # Generate a unique ID for this query

    relevant_chunks = await hybrid_search(db, payload.query, top_k=4)

    if not relevant_chunks:
        return {"answer": "I could not find any relevant information in the documents.", "sources": []}

    context_texts = [c.chunk_text for c in relevant_chunks]

    # Pass the trace_id to the generation function
    answer = await generate_answer(payload.query, context_texts, trace_id)

    sources = [{"document": c.document_name,
                "text_preview": c.chunk_text[:100] + "..."} for c in relevant_chunks]

    return {
        "query": payload.query,
        "answer": answer,
        "sources": sources,
        "langfuse_trace_url": f"{settings.langfuse_host}/trace/{trace_id}"
    }


@app.get("/health")
async def health_check():
    return {"status": "operational", "model": settings.embedding_model}


# Initialize Langfuse client
langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host
)

# --- DIAGNOSTIC CHECK ---
try:
    langfuse.auth_check()
    print("✅ SUCCESS: Langfuse SDK connected to server successfully!")
except Exception as e:
    print(f"❌ CRITICAL: Langfuse connection failed. Error: {e}")
    print("Check your .env keys and ensure the Docker container is running.")
# ------------------------

# <-- This decorator magically traces the function
# Initialize Langfuse client at the module level


async def generate_answer(query: str, context_chunks: list[str], trace_id: str) -> str:
    """Calls Ollama to generate an answer based on the retrieved context, traced via Langfuse."""
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a precise AI assistant. Answer the user's question using ONLY the provided context. 
If the answer is not explicitly stated in the context, reply with: "I do not have enough information in the provided context to answer that."

Context:
{context}

Question: {query}
"""

    # 1. Initialize the Langfuse Trace
    trace = langfuse.trace(
        id=trace_id,
        name="rag-query",
        input={"query": query, "context_chunks": context_chunks},
        metadata={"model": "llama3.1:8b",
                  "embedding_model": settings.embedding_model}
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # 2. Make the actual API call to Ollama
            ollama_response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            ollama_response.raise_for_status()
            answer_text = ollama_response.json()["message"]["content"]

            # 3. Log the successful generation to Langfuse
            trace.generation(
                name="llama3-generation",
                model="llama3.1:8b",
                input=prompt,
                output=answer_text
            )

            return answer_text

        except httpx.HTTPStatusError as e:
            # Log the error to Langfuse so you can debug it later
            trace.update(output={"error": f"HTTP Error: {e.response.text}"})
            raise HTTPException(
                status_code=500, detail=f"LLM API error: {e.response.text}")
        except Exception as e:
            # Log the error to Langfuse
            trace.update(output={"error": str(e)})
            raise HTTPException(
                status_code=500, detail=f"LLM Generation failed: {str(e)}")

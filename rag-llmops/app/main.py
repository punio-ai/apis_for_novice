from app.processing import extract_text_from_pdf, chunk_text
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi import HTTPException
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from app.database import init_db, get_db
from app.models import DocumentChunk
from app.config import settings
from pydantic import BaseModel

app = FastAPI(title="Production RAG LLMOps API (Ollama)")


@app.on_event("startup")
async def startup():
    await init_db()

# --- OLLAMA HELPER ---


async def get_embedding(text: str) -> list[float]:
    """Fetches embedding from local Ollama instance asynchronously."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": settings.embedding_model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=500, detail=f"Ollama API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to generate embedding: {str(e)}")

# --- ENDPOINTS ---


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Fetches embeddings for a batch of texts from local Ollama asynchronously."""
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Using the newer /api/embed endpoint which supports batch inputs
            response = await client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={
                    "model": settings.embedding_model,
                    "input": texts
                }
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=500, detail=f"Ollama API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to generate embeddings: {str(e)}")


class IngestRequest(BaseModel):
    document_name: str
    text: str


@app.post("/ingest")
async def ingest_document(payload: IngestRequest, db: AsyncSession = Depends(get_db)):
    # 1. Generate real embedding via Ollama
    embedding_vector = await get_embedding(payload.text)

    # 2. Store in pgvector
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


# --- NEW PRODUCTION ENDPOINT ---
@app.post("/ingest-file")
async def ingest_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are currently supported.")

    # 1. Read file asynchronously
    file_bytes = await file.read()

    # 2. Extract text
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text.strip():
        raise HTTPException(
            status_code=400, detail="Could not extract text from PDF. Is it scanned/image-based?")

    # 3. Chunk the text
    chunks = chunk_text(raw_text)
    if not chunks:
        raise HTTPException(
            status_code=400, detail="No chunks generated from text.")

    # 4. Batch generate embeddings (The LLMOps Performance Trick)
    # We process in batches of 20 to avoid overwhelming Ollama's memory
    all_embeddings = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = await get_embeddings_batch(batch_chunks)
        all_embeddings.extend(batch_embeddings)

    # 5. Bulk insert into PostgreSQL
    db_objects = [
        DocumentChunk(
            document_name=file.filename,
            chunk_text=chunk_text_item,
            embedding=embedding_vector
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


@app.get("/health")
async def health_check():
    return {"status": "operational", "model": settings.embedding_model}

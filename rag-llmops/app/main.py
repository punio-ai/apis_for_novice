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


@app.get("/health")
async def health_check():
    return {"status": "operational", "model": settings.embedding_model}

import httpx
from fastapi import HTTPException
from app.config import settings


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


async def get_embedding(text: str) -> list[float]:
    """Fetches a single embedding from local Ollama instance asynchronously."""
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

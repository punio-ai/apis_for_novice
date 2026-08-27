import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_create_and_retrieve_note():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create
        response = await client.post("/api/v1/notes", json={
            "title": "Test Note",
            "content": "This is a test about RAG evaluation",
            "tags": ["test"]
        })
        assert response.status_code == 201
        note_id = response.json()["id"]

        # Retrieve
        response = await client.get(f"/api/v1/notes/{note_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Test Note"


@pytest.mark.asyncio
async def test_search():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/v1/notes", json={
            "title": "GraphRAG",
            "content": "Knowledge graphs enable multi-hop reasoning",
            "tags": ["graph"]
        })
        response = await client.post("/api/v1/search", json={
            "query": "multi-hop",
            "limit": 5
        })
        assert response.status_code == 200
        assert len(response.json()) >= 1

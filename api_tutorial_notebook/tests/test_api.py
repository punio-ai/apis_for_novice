import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from langchain_core.documents import Document
from app.main import app

HEADERS = {"x-api-key": "super-secret-key-2026"}


@pytest.mark.asyncio
async def test_summarize_note_background_task():
    """
    Tests that the summarize endpoint triggers the background task 
    and returns 202 Accepted, without actually calling the real LLM.
    """
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create a note to summarize
        create_response = await client.post("/api/v1/notes", headers=HEADERS, json={
            "title": "Testing Mocks",
            "content": "Mocking is a technique used to isolate the code being tested.",
            "tags": ["testing"]
        })
        assert create_response.status_code == 201
        note_id = create_response.json()["id"]
        note_content = create_response.json()["content"]

        # 2. MOCK the LLM service function where it is USED (in app.routes)
        with patch("app.routes.generate_summary", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Mocked Summary: 1. Mocking is great."

            # 3. Trigger the summarize endpoint
            summarize_response = await client.post(f"/api/v1/notes/{note_id}/summarize", headers=HEADERS)

            # 4. Assert the API responded correctly
            assert summarize_response.status_code == 202
            assert summarize_response.json()["status"] == "processing"

            # 5. Assert the mock was called with the correct content
            mock_llm.assert_awaited_once_with(note_content)


@pytest.mark.asyncio
async def test_search_endpoint_with_mocked_chromadb():
    """
    Tests the /search endpoint by mocking ChromaDB and the background embedding task.
    This ensures the test runs in milliseconds without needing Ollama or a real DB.
    """
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Mock the background embedding task so note creation is instant
        # and doesn't try to call Ollama during the test.
        with patch("app.routes.background_embedding_task", new_callable=AsyncMock) as mock_bg_task:

            # Create a note to search for
            create_response = await client.post("/api/v1/notes", headers=HEADERS, json={
                "title": "Mocked Vector Note",
                "content": "This note is for testing semantic search.",
                "tags": ["test"]
            })
            assert create_response.status_code == 201
            note_id = create_response.json()["id"]

            # 2. Mock the ChromaDB similarity search
            # We patch it where it is USED (in app.routes)
            mock_doc = Document(
                page_content="This note is for testing semantic search.",
                metadata={"note_id": note_id, "title": "Mocked Vector Note"}
            )
            # ChromaDB returns (Document, score) tuples.
            mock_search_results = [(mock_doc, 0.15)]

            with patch("app.routes.vector_store.similarity_search_with_score", return_value=mock_search_results):

                # 3. Call the search endpoint
                search_response = await client.post("/api/v1/search", headers=HEADERS, json={
                    "query": "semantic testing",
                    "limit": 3
                })

                # 4. Assert the response
                assert search_response.status_code == 200
                results = search_response.json()
                assert len(results) == 1
                assert results[0]["id"] == note_id
                assert results[0]["title"] == "Mocked Vector Note"

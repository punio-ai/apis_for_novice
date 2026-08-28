import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from app.main import app

# Standard headers for authenticated requests
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
            "content": "Mocking is a technique used to isolate the code being tested from external dependencies.",
            "tags": ["testing", "mocks"]
        })
        assert create_response.status_code == 201
        note_id = create_response.json()["id"]
        note_content = create_response.json()["content"]

        # 2. MOCK the LLM service function
        # We patch it where it is USED (in app.routes), not where it is defined.
        with patch("app.routes.generate_summary", new_callable=AsyncMock) as mock_llm:

            # Configure the mock to return a fake summary instantly
            mock_llm.return_value = "Mocked Summary: 1. Mocking is great. 2. It is fast. 3. It is reliable."

            # 3. Trigger the summarize endpoint
            summarize_response = await client.post(f"/api/v1/notes/{note_id}/summarize", headers=HEADERS)

            # 4. Assert the API responded correctly
            assert summarize_response.status_code == 202
            assert summarize_response.json()["status"] == "processing"
            assert summarize_response.json()["note_id"] == note_id

            # 5. Assert the mock was called exactly once with the correct content
            # 5. Assert the mock was called exactly once with the correct content
            mock_llm.assert_awaited_once_with(note_content)

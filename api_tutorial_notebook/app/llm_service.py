import httpx
import logging

logger = logging.getLogger(__name__)

# UPGRADE: Use the /api/chat endpoint for instruct/chat models
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:latest"


async def generate_summary(text: str) -> str:
    """
    Calls the local Ollama API using the modern /api/chat format 
    to generate a 3-bullet-point summary.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise technical assistant. Summarize the provided text into exactly 3 concise bullet points. Do not add conversational filler or introductory text."
            },
            {
                "role": "user",
                "content": f"Summarize the following text:\n\n{text}"
            }
        ],
        "stream": False  # We want the full response at once for a background task
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()

            result = response.json()
            # The /api/chat endpoint returns the text inside message.content
            return result.get("message", {}).get("content", "No summary generated.")

    except httpx.ConnectError:
        logger.error("Failed to connect to Ollama. Is 'ollama serve' running?")
        return "Error: Could not connect to local Ollama server."
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama API returned an error: {e}")
        return f"Error: LLM failed with status {e.response.status_code}"
    except Exception as e:
        logger.error(f"Unexpected error during LLM call: {e}")
        return "Error: An unexpected error occurred during summarization."

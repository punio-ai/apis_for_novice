import httpx
import logging

logger = logging.getLogger(__name__)

# Point to your local Ollama instance
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"


async def generate_summary(text: str) -> str:
    """
    Calls the local Ollama API to generate a 3-bullet-point summary.
    """
    prompt = f"Summarize the following text in exactly 3 concise bullet points:\n\n{text}"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False  # We want the full response at once for a background task
    }

    try:
        # Use a timeout because local LLMs can sometimes hang or take time to load into memory
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()  # Raises an error for 4xx/5xx status codes

            result = response.json()
            return result.get("response", "No summary generated.")

    except httpx.ConnectError:
        logger.error("Failed to connect to Ollama. Is 'ollama serve' running?")
        return "Error: Could not connect to local Ollama server."
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama API returned an error: {e}")
        return f"Error: LLM failed with status {e.response.status_code}"
    except Exception as e:
        logger.error(f"Unexpected error during LLM call: {e}")
        return "Error: An unexpected error occurred during summarization."

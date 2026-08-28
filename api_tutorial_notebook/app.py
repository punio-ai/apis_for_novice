import gradio as gr
import httpx

# Point this to your running FastAPI server
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
API_KEY = "super-secret-key-2026"
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}


async def chat_with_knowledge(query: str, history: list):
    """Main function that handles the UI interaction."""
    if not query.strip():
        yield history, "Please enter a query."
        return

    # 1. Add user message to history
    history.append({"role": "user", "content": query})
    yield history, "🔍 Searching knowledge base..."

    try:
        # 2. Call the Semantic Search Endpoint
        async with httpx.AsyncClient() as client:
            search_response = await client.post(
                f"{API_BASE_URL}/search",
                headers=HEADERS,
                json={"query": query, "limit": 3}
            )

            if search_response.status_code != 200:
                yield history, f"❌ API Error: {search_response.text}"
                return

            results = search_response.json()

            if not results:
                history.append(
                    {"role": "assistant", "content": "I couldn't find any relevant notes in the database."})
                yield history, "No results found."
                return

            # 3. Format the retrieved context
            formatted_results = "**📚 Retrieved Notes:**\n" + \
                "\n".join([f"- **{r['title']}**" for r in results])

            # 4. Trigger Summarization Endpoint for the best match
            best_note_id = results[0]["id"]

            final_answer = f"{formatted_results}\n\n**🤖 AI Action:**\nTriggered background summarization for note #{best_note_id}. Check your FastAPI terminal for the LLM output!"

            history.append({"role": "assistant", "content": final_answer})

            # Fire and forget: trigger the summary in the background
            await client.post(
                f"{API_BASE_URL}/notes/{best_note_id}/summarize",
                headers=HEADERS
            )

            yield history, "✅ Done! Summary is generating in the background."

    except httpx.ConnectError:
        history.append(
            {"role": "assistant", "content": "❌ Could not connect to the API. Is the FastAPI server running?"})
        yield history, "Connection failed."

# ==========================================
# GRADIO UI SETUP
# ==========================================
# FIX 1: Removed 'theme' from Blocks constructor
with gr.Blocks(title="2026 Local RAG Dashboard") as demo:
    gr.Markdown("# 🧠 Local Knowledge Base RAG Dashboard")
    gr.Markdown("Powered by FastAPI, ChromaDB, and Ollama. 100% Local.")

    with gr.Row():
        with gr.Column(scale=2):
            # FIX 2: Removed type="messages" (it is the default in modern Gradio)
            chatbot = gr.Chatbot(height=400, label="Chat Interface")
            with gr.Row():
                msg = gr.Textbox(label="Ask a question about your notes...", scale=4,
                                 placeholder="e.g., How do I store AI embeddings locally?")
                submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 🔍 System Status")
            status_box = gr.Textbox(
                label="Live Telemetry", value="Ready", interactive=False)
            gr.Markdown("---")
            gr.Markdown("### 🛠️ Tech Stack")
            gr.Markdown("- **Backend:** FastAPI (Async)")
            gr.Markdown("- **Vector DB:** ChromaDB")
            gr.Markdown("- **Embeddings:** `nomic-embed-text`")
            gr.Markdown("- **LLM:** `llama3.2:latest` (Ollama)")

    # Event Handlers
    msg.submit(chat_with_knowledge, [msg, chatbot], [chatbot, status_box])
    submit_btn.click(chat_with_knowledge, [
                     msg, chatbot], [chatbot, status_box])

if __name__ == "__main__":
    # FIX 3: Moved 'theme' to the launch() method as the warning instructed
    demo.launch(server_name="127.0.0.1",
                server_port=7860, theme=gr.themes.Soft())

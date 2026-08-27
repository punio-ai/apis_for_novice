## Week 1 — Aug 2026
- Built my first FastAPI app with async database
- Learned Pydantic V2 validation (Field, model_config)
- Dockerized it and ran it locally
- Wrote async tests with pytest
- Broke: Forgot `__init__.py` in app folder, spent 30 min debugging import errors
- Learned: Python needs `__init__.py` to treat folders as packages
- Feeling: Actually built something real. Small but real.
## 🚀 How to Run Locally

1. Clone the repository:
bash
   git clone https://github.com/punio-ai/apis_for_novice.git

   cd apis_for_novice


2. Build and run with Docker:
   ```bash
   docker build -t knowledge-api .
   docker run -p 8000:8000 knowledge-api

3. Open your browser to http://localhost:8000/docs
   
---

### What’s Next? (Choose Your Path for Week 2)

You have the skeleton. Now we start injecting the "AI" into the AI Engineer title. For your next practice session, choose **one** of these paths:

**Path A: The RAG Foundation (Recommended)**  
Replace the simple keyword search in `routes.py` with a real **Vector Search**. We will add `langchain` and `chromadb` to your `requirements.txt`, create an embedding function, and make the `/search` endpoint actually perform semantic similarity search on the `content` field of your notes.

**Path B: The Agentic Tool**  
Add a new endpoint `/api/v1/summarize` that takes a `note_id`, fetches that note from the database, and sends it to your local `ollama` instance (using `langchain-ollama`) to generate a 3-bullet-point summary, returning it to the user.

**Path C: The Deployment Test**  
Try deploying this exact Docker container to a free tier service like **Render** or **Railway** just to see what it takes to get it off your localhost and onto the public internet.

Which path excites you the most for your next session? Tell me, and I will give you the exact code and steps to build it. You are doing incredible work. Keep shipping.

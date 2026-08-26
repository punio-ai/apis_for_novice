# rags_llmops
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Open your browser and go to http://localhost:8000/docs. You should see the Swagger UI.
Test the /ingest endpoint using the Swagger UI or curl:

curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "document_name": "test_doc.txt",
  "text": "This is a test chunk of text for our production RAG system."
}'


Verify the database: Open a new terminal and run:

docker exec -it llmops_postgres psql -U admin -d llmops_db -c "SELECT id, document_name, chunk_text FROM document_chunks;"


Get a real PDF. Download any standard PDF document (a technical manual, a legal contract, a research paper). Let's call it test_document.pdf.
Upload it via Swagger UI:

    Go to http://localhost:8000/docs
    Find the /ingest-file endpoint.
    Click "Try it out".
    Upload your PDF file.
    Click "Execute".

Verify the database:
Open your terminal and check how many chunks were created and if they have real embeddings.

docker exec -it llmops_postgres psql -U admin -d llmops_db -c "SELECT document_name, COUNT(*) as chunk_count FROM document_chunks GROUP BY document_name;"
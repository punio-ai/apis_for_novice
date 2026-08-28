import asyncio
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import async_session, NoteDB
from app.models import NoteCreate, NoteResponse, SearchQuery
from app.dependencies import pagination, get_api_key
from app.llm_service import generate_summary
from app.vector_store import vector_store  # <-- Import our new vector store

router = APIRouter()


async def get_db():
    async with async_session() as session:
        yield session

# 1. REAL Background Task for Embeddings


async def background_embedding_task(note_id: int, content: str, title: str):
    print(f"⏳ [Background] Generating embedding for Note {note_id}...")
    try:
        # We use the note_id as the ChromaDB document ID for easy reference later
        doc_id = str(note_id)
        metadata = {"note_id": note_id, "title": title}

        # NOTE: LangChain's add_texts is synchronous.
        # We use asyncio.to_thread to run it without blocking the async event loop.
        # This is a PRODUCTION-GRADE async pattern.
        await asyncio.to_thread(
            vector_store.add_texts,
            texts=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(
            f"✅ [Background] Embedding for Note {note_id} saved to ChromaDB.")
    except Exception as e:
        print(f"❌ [Background] Failed to embed Note {note_id}: {e}")


@router.post("/notes", response_model=NoteResponse, status_code=201)
async def create_note(
    note: NoteCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    db_note = NoteDB(title=note.title, content=note.content, tags=note.tags)
    db.add(db_note)
    await db.commit()
    await db.refresh(db_note)

    # Queue the REAL embedding task (passing 3 arguments)
    background_tasks.add_task(
        background_embedding_task, db_note.id, db_note.content, db_note.title)

    return db_note

# 2. REAL Semantic Search Endpoint


@router.post("/search", response_model=list[NoteResponse])
async def search_notes(
    query: SearchQuery,
    db: AsyncSession = Depends(get_db)
):
    print(f"🔍 Semantic search for: '{query.query}' (Limit: {query.limit})")

    # 1. Query ChromaDB for similar documents
    # similarity_search_with_score returns a list of tuples: (Document, score)
    results = await asyncio.to_thread(
        vector_store.similarity_search_with_score,
        query=query.query,
        k=query.limit
    )

    if not results:
        return []

    # 2. Extract the note_ids from the ChromaDB metadata
    # We sort by score (lower score = more similar in ChromaDB cosine distance)
    sorted_results = sorted(results, key=lambda x: x[1])
    note_ids = [int(doc.metadata["note_id"]) for doc, score in sorted_results]

    # 3. Fetch the full Note objects from SQLite to match our NoteResponse Pydantic model
    result = await db.execute(select(NoteDB).where(NoteDB.id.in_(note_ids)))
    notes = result.scalars().all()

    # 4. Return the notes (ordered by semantic relevance)
    # We create a mapping to easily sort the SQLAlchemy results by the ChromaDB order
    note_map = {note.id: note for note in notes}
    return [note_map[nid] for nid in note_ids if nid in note_map]


@router.post("/notes/{note_id}/summarize", status_code=202)
async def trigger_summarize(
    note_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    # First, verify the note exists
    result = await db.execute(select(NoteDB).where(NoteDB.id == note_id))
    note = result.scalar_one_or_none()

    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Queue the task. FastAPI will execute this AFTER returning the HTTP response.
    background_tasks.add_task(summarize_note_task, note_id, note.content)

    return {
        "message": "Summarization started in the background",
        "note_id": note_id,
        "status": "processing"
    }


# Simulated heavy AI task (e.g., generating embeddings)
# 1. REAL Background Task for Embeddings
async def background_embedding_task(note_id: int, content: str, title: str):
    print(f"⏳ [Background] Generating embedding for Note {note_id}...")
    try:
        # We use the note_id as the ChromaDB document ID for easy reference later
        doc_id = str(note_id)
        metadata = {"note_id": note_id, "title": title}

        # NOTE: LangChain's add_texts is synchronous.
        # We use asyncio.to_thread to run it without blocking the async event loop.
        await asyncio.to_thread(
            vector_store.add_texts,
            texts=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(
            f"✅ [Background] Embedding for Note {note_id} saved to ChromaDB.")
    except Exception as e:
        print(f"❌ [Background] Failed to embed Note {note_id}: {e}")


@router.post("/notes", response_model=NoteResponse, status_code=201)
async def create_note(
    note: NoteCreate,
    background_tasks: BackgroundTasks,  # Injected by FastAPI
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)  # Enforces auth
):
    db_note = NoteDB(title=note.title, content=note.content, tags=note.tags)
    db.add(db_note)
    await db.commit()
    await db.refresh(db_note)

    # Queue the heavy task to run AFTER the response is sent to the user
    background_tasks.add_task(
        background_embedding_task, db_note.id, db_note.content)

    return db_note


@router.get("/notes", response_model=list[NoteResponse])
async def list_notes(
    page: dict = Depends(pagination),  # Inject our reusable dependency
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(NoteDB).offset(page["skip"]).limit(page["limit"]))
    return result.scalars().all()

# ... (keep your existing get_note and search_notes endpoints)


@router.post("/search", response_model=list[NoteResponse])
async def search_notes(query: SearchQuery, db: AsyncSession = Depends(get_db)):
    """Simple keyword search. In Month 2, this becomes your RAG retriever."""
    search_term = f"%{query.query}%"
    result = await db.execute(
        select(NoteDB)
        .where(NoteDB.content.ilike(search_term) | NoteDB.title.ilike(search_term))
        .limit(query.limit)
    )
    return result.scalars().all()

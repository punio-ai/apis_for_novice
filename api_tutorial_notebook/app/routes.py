import asyncio
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import async_session, NoteDB
from app.models import NoteCreate, NoteResponse, SearchQuery
from app.dependencies import pagination, get_api_key
from app.llm_service import generate_summary
from app.vector_store import vector_store

router = APIRouter()


async def get_db():
    async with async_session() as session:
        yield session

# ==========================================
# BACKGROUND TASKS
# ==========================================


async def background_embedding_task(note_id: int, content: str, title: str):
    print(f"⏳ [Background] Generating embedding for Note {note_id}...")
    try:
        doc_id = str(note_id)
        metadata = {"note_id": note_id, "title": title}

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


async def summarize_note_task(note_id: int, content: str):
    print(f"⏳ [Background Task] Starting LLM summary for Note {note_id}...")
    summary = await generate_summary(content)
    print(
        f"✅ [Background Task] Summary for Note {note_id} completed:\n{summary}")

# ==========================================
# ROUTES
# ==========================================


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


@router.get("/notes", response_model=list[NoteResponse])
async def list_notes(
    page: dict = Depends(pagination),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(NoteDB).offset(page["skip"]).limit(page["limit"]))
    return result.scalars().all()


@router.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(NoteDB).where(NoteDB.id == note_id))
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.post("/search", response_model=list[NoteResponse])
async def search_notes(
    query: SearchQuery,
    db: AsyncSession = Depends(get_db)
):
    print(f"🔍 Semantic search for: '{query.query}' (Limit: {query.limit})")

    # 1. Query ChromaDB for similar documents
    results = await asyncio.to_thread(
        vector_store.similarity_search_with_score,
        query=query.query,
        k=query.limit
    )

    if not results:
        return []

    # 2. Extract note_ids and sort by score (lower score = more similar)
    sorted_results = sorted(results, key=lambda x: x[1])
    note_ids = [int(doc.metadata["note_id"]) for doc, score in sorted_results]

    # 3. Fetch full Note objects from SQLite
    result = await db.execute(select(NoteDB).where(NoteDB.id.in_(note_ids)))
    notes = result.scalars().all()

    # 4. Return notes ordered by semantic relevance
    note_map = {note.id: note for note in notes}
    return [note_map[nid] for nid in note_ids if nid in note_map]


@router.post("/notes/{note_id}/summarize", status_code=202)
async def trigger_summarize(
    note_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    result = await db.execute(select(NoteDB).where(NoteDB.id == note_id))
    note = result.scalar_one_or_none()

    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Queue the summarization task
    background_tasks.add_task(summarize_note_task, note_id, note.content)

    return {
        "message": "Summarization started in the background",
        "note_id": note_id,
        "status": "processing"
    }

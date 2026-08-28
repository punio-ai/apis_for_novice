from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import async_session, NoteDB
from app.models import NoteCreate, NoteResponse, SearchQuery
from app.dependencies import pagination, get_api_key
from app.llm_service import generate_summary  # <-- Import our new service
import asyncio

router = APIRouter()


async def get_db():
    async with async_session() as session:
        yield session

# 1. Define the Background Task


async def summarize_note_task(note_id: int, content: str):
    print(f"⏳ [Background Task] Starting LLM summary for Note {note_id}...")

    # Call the LLM service
    summary = await generate_summary(content)

    print(f"✅ [Background Task] Summary for Note {note_id} completed:")
    print("-" * 40)
    print(summary)
    print("-" * 40)

    # PRO TIP: In a real app, you would open a new DB session here
    # and update the NoteDB with `note.summary = summary`

# 2. Define the Route that triggers it


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


async def background_embedding_task(note_id: int, content: str):
    print(
        f"⏳ [Background] Starting embedding generation for Note {note_id}...")
    await asyncio.sleep(3)  # Simulate 3 seconds of LLM/Embedding API latency
    print(f"✅ [Background] Embedding for Note {note_id} completed and saved.")


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

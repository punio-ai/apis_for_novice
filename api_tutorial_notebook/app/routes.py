from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import async_session, NoteDB
from app.models import NoteCreate, NoteResponse, SearchQuery

router = APIRouter()


async def get_db():
    async with async_session() as session:
        yield session


@router.post("/notes", response_model=NoteResponse, status_code=201)
async def create_note(note: NoteCreate, db: AsyncSession = Depends(get_db)):
    db_note = NoteDB(title=note.title, content=note.content, tags=note.tags)
    db.add(db_note)
    await db.commit()
    await db.refresh(db_note)
    return db_note


@router.get("/notes", response_model=list[NoteResponse])
async def list_notes(skip: int = 0, limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(NoteDB).offset(skip).limit(limit))
    return result.scalars().all()


@router.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(NoteDB).where(NoteDB.id == note_id))
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


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

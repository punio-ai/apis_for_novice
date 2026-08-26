from sqlalchemy import Column, Integer, String, Text, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped
from pgvector.sqlalchemy import Vector
from app.database import Base
from app.config import settings


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    document_name: Mapped[str] = Column(String, index=True)
    chunk_text: Mapped[str] = Column(Text)
    embedding = Column(Vector(settings.embedding_dimensions))

    # NEW: Full Text Search column for Hybrid Search
    text_search = Column(TSVECTOR)

    # NEW: GIN index for fast keyword lookups
    __table_args__ = (
        Index('ix_document_chunks_text_search',
              text_search, postgresql_using='gin'),
    )

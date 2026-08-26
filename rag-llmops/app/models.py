from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import Mapped
from pgvector.sqlalchemy import Vector
from app.database import Base
from app.config import settings


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    document_name: Mapped[str] = Column(String, index=True)
    chunk_text: Mapped[str] = Column(Text)
    # Dynamically set dimensions based on the model we are using
    embedding = Column(Vector(settings.embedding_dimensions))

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# What the user sends to create a note


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200,
                       examples=["How Hybrid Search Works"])
    content: str = Field(..., min_length=1, examples=[
                         "BM25 + Dense Vectors fused via RRF..."])
    tags: list[str] = Field(default_factory=list, examples=[
                            ["rag", "search", "bm25"]])

# What we return to the user


class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    tags: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}  # Pydantic V2 syntax

# What the user sends to search


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, examples=["hybrid search"])
    limit: int = Field(default=5, ge=1, le=50)

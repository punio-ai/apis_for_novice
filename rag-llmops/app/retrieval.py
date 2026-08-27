from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models import DocumentChunk
from app.ollama_client import get_embeddings_batch

# Lazy-load FlashRank to prevent import crashes if HuggingFace is blocked/unreachable
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from flashrank import Ranker
            print("✅ FlashRank loaded successfully.")
            _reranker = Ranker()
        except Exception as e:
            print(
                f"⚠️ Warning: FlashRank failed to load (Network/SSL issue). Falling back to RRF only. Error: {e}")
            _reranker = False  # Use False to indicate it's permanently failed for this session
    return _reranker


async def hybrid_search(db: AsyncSession, query: str, top_k: int = 5):
    """Executes Hybrid Search (Semantic + Keyword) and Re-ranks the results."""

    # 1. Get the query embedding for semantic search
    query_embedding = (await get_embeddings_batch([query]))[0]

    # 2. Semantic Search (Top 20)
    semantic_stmt = (
        select(DocumentChunk)
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(20)
    )
    semantic_result = await db.execute(semantic_stmt)
    semantic_chunks = semantic_result.scalars().all()

    # 3. Keyword Search (Top 20)
    keyword_stmt = (
        select(DocumentChunk)
        .filter(DocumentChunk.text_search.op('@@')(func.plainto_tsquery('english', query)))
        .limit(20)
    )
    keyword_result = await db.execute(keyword_stmt)
    keyword_chunks = keyword_result.scalars().all()

    # 4. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    k = 60  # Standard RRF constant

    for rank, chunk in enumerate(semantic_chunks):
        rrf_scores[chunk.id] = rrf_scores.get(
            chunk.id, 0) + (1 / (k + rank + 1))
    for rank, chunk in enumerate(keyword_chunks):
        rrf_scores[chunk.id] = rrf_scores.get(
            chunk.id, 0) + (1 / (k + rank + 1))

    # Sort by RRF score and fetch the top 15 candidates
    sorted_ids = sorted(rrf_scores.keys(),
                        key=lambda x: rrf_scores[x], reverse=True)[:15]

    final_candidates_stmt = select(DocumentChunk).filter(
        DocumentChunk.id.in_(sorted_ids))
    final_candidates_result = await db.execute(final_candidates_stmt)
    candidates = final_candidates_result.scalars().all()

    chunk_map = {c.id: c for c in candidates}
    sorted_candidates = [chunk_map[c_id]
                         for c_id in sorted_ids if c_id in chunk_map]

    if not sorted_candidates:
        return []

    # 5. Re-ranking using FlashRank (with graceful degradation)
    reranker = get_reranker()
    if reranker:
        from flashrank import RerankRequest
        passages = [{"id": str(c.id), "text": c.chunk_text}
                    for c in sorted_candidates]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = reranker.rerank(rerank_request)

        final_chunks = []
        for res in reranked_results[:top_k]:
            chunk_id = int(res["id"])
            for c in sorted_candidates:
                if c.id == chunk_id:
                    final_chunks.append(c)
                    break
        return final_chunks
    else:
        # Fallback: just return the top_k from RRF
        return sorted_candidates[:top_k]

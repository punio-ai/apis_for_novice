from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models import DocumentChunk
# We will reuse our Ollama embedding helper
from app.ollama_client import get_embeddings_batch
from flashrank import Ranker, RerankRequest

# Initialize the re-ranker (downloads a tiny ~30MB model on first run)
reranker = Ranker()


async def hybrid_search(db: AsyncSession, query: str, top_k: int = 5):
    """
    Executes Hybrid Search (Semantic + Keyword) and Re-ranks the results.
    """
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
    # Combines the two lists, giving higher scores to chunks that appear in both or rank highly
    rrf_scores = {}
    k = 60  # Standard RRF constant

    for rank, chunk in enumerate(semantic_chunks):
        rrf_scores[chunk.id] = rrf_scores.get(
            chunk.id, 0) + (1 / (k + rank + 1))
    for rank, chunk in enumerate(keyword_chunks):
        rrf_scores[chunk.id] = rrf_scores.get(
            chunk.id, 0) + (1 / (k + rank + 1))

    # Sort by RRF score and fetch the top 15 candidates for re-ranking
    sorted_ids = sorted(rrf_scores.keys(),
                        key=lambda x: rrf_scores[x], reverse=True)[:15]

    # Fetch the actual chunk objects for the top 15
    final_candidates_stmt = select(DocumentChunk).filter(
        DocumentChunk.id.in_(sorted_ids))
    final_candidates_result = await db.execute(final_candidates_stmt)
    candidates = final_candidates_result.scalars().all()

    # Map them back to a dictionary for easy access
    chunk_map = {c.id: c for c in candidates}
    sorted_candidates = [chunk_map[c_id]
                         for c_id in sorted_ids if c_id in chunk_map]

    # 5. Re-ranking using FlashRank
    if not sorted_candidates:
        return []

    passages = [{"id": str(c.id), "text": c.chunk_text}
                for c in sorted_candidates]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = reranker.rerank(rerank_request)

    # Return the top K final chunks
    final_chunks = []
    for res in reranked_results[:top_k]:
        chunk_id = int(res["id"])
        final_chunks.append(chunk_map[chunk_id])

    return final_chunks

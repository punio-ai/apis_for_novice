import asyncio
import json
import re
import httpx
from app.config import settings
from app.retrieval import hybrid_search
from app.main import generate_answer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 1. Define "Golden" test cases based on your specific PDF
EVAL_DATASET = [
    {
        "input": "What is the passing score for the exam?",
        "expected_keywords": ["750"]  # Verify this matches your PDF
    },
    {
        "input": "Who is this exam intended for?",
        "expected_keywords": ["generative ai developer", "genai developer"]
    },
    {
        "input": "What is the exam code?",
        "expected_keywords": ["aip-c01"]
    }
]


async def run_standalone_query(query: str) -> dict:
    """Standalone query function that creates its own DB session."""
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        relevant_chunks = await hybrid_search(db, query, top_k=4)
        if not relevant_chunks:
            return {"answer": "No relevant information found.", "contexts": []}

        context_texts = [c.chunk_text for c in relevant_chunks]
        import uuid
        answer = await generate_answer(query, context_texts, str(uuid.uuid4()))
        return {"answer": answer, "contexts": context_texts}


async def evaluate_with_local_llm(query: str, answer: str, contexts: list[str]) -> dict:
    """Uses local Ollama native JSON mode to score Faithfulness and Relevance quickly."""
    context_str = "\n---\n".join(contexts)

    prompt = f"""You are a strict RAG evaluator. Score the following answer from 0 to 10 on two metrics:
1. FAITHFULNESS: Does the answer rely ONLY on the provided context? (10 = perfectly grounded, 0 = hallucinated)
2. RELEVANCE: Does the answer directly address the user's query? (10 = perfect, 0 = irrelevant)

Query: {query}
Context: {context_str}
Answer: {answer}

Return ONLY a valid JSON object with two integer keys: "faithfulness" and "relevance". 
Example: {{"faithfulness": 9, "relevance": 10}}"""

    # Increased timeout to 120s in case the model needs time to load into RAM
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json"
                }
            )

            # DEBUG: Print exactly what Ollama is returning
            print(f"   🔍 Raw HTTP Status: {response.status_code}")
            print(f"   🔍 Raw Response Body: {response.text[:200]}...")

            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")

            if not content:
                print(f"   ️ Empty content from Ollama. Full response: {data}")
                return {"faithfulness": 0, "relevance": 0}

            # Strip markdown backticks if the LLM adds them
            content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
            content = content.strip()

            return json.loads(content)

        except Exception as e:
            print(f"   ⚠️ Evaluation Error: {type(e).__name__} - {str(e)}")
            return {"faithfulness": 0, "relevance": 0, "error": str(e)}


async def run_evaluation():
    print("🔍 Starting Lightweight Custom RAG Evaluation...")
    results = []

    for item in EVAL_DATASET:
        print(f"\n   ⚙️ Querying: '{item['input']}'")
        rag_result = await run_standalone_query(item["input"])
        answer = rag_result["answer"]
        contexts = rag_result["contexts"]

        print(f"   ⚖️ Evaluating with local LLM judge...")
        scores = await evaluate_with_local_llm(item["input"], answer, contexts)

        # Simple keyword check for baseline precision
        keyword_hit = any(kw.lower() in answer.lower()
                          for kw in item["expected_keywords"])

        results.append({
            "Query": item["input"],
            "Answer": answer[:50] + "..." if len(answer) > 50 else answer,
            "Faithfulness": scores.get("faithfulness", 0),
            "Relevance": scores.get("relevance", 0),
            "Keyword Match": "✅ PASS" if keyword_hit else "❌ FAIL"
        })
        print(f"   ✅ Done.")

    # Print beautiful, portfolio-ready table
    print("\n" + "="*85)
    print("📈 RAG SYSTEM EVALUATION RESULTS")
    print("="*85)
    print(f"{'Query':<35} | {'Faithfulness':<12} | {'Relevance':<12} | {'Keyword'}")
    print("-"*85)
    for r in results:
        print(
            f"{r['Query']:<35} | {r['Faithfulness']:<12} | {r['Relevance']:<12} | {r['Keyword Match']}")
    print("="*85)

    avg_faith = sum(r["Faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["Relevance"] for r in results) / len(results)
    print(f"🏆 AVERAGE FAITHFULNESS: {avg_faith:.1f}/10")
    print(f"🏆 AVERAGE RELEVANCE:   {avg_rel:.1f}/10")
    print("="*85)

if __name__ == "__main__":
    asyncio.run(run_evaluation())

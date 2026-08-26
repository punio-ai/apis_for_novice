from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import asyncio

# You will hardcode a few "Golden" questions and ground truth answers for your specific PDF
eval_data = {
    "question": ["Who is this exam meant for?", "What is the exam code?"],
    "answer": [],  # Your RAG system will fill this
    "contexts": [],  # Your RAG system will fill this
    "ground_truth": ["Individuals who perform a GenAI developer role.", "AIP-C01"]
}


async def run_evaluation(rag_results: list[dict]):
    # Populate the eval_data with your system's actual outputs
    eval_data["answer"] = [r["answer"] for r in rag_results]
    eval_data["contexts"] = [[src["text_preview"]
                              for src in r["sources"]] for r in rag_results]

    dataset = Dataset.from_dict(eval_data)

    # Run evaluation (this will use a local LLM or OpenAI as the "judge")
    # For now, we will configure it to use your local Ollama model as the judge
    from ragas.llms import LangchainLLMWrapper
    from langchain_ollama import ChatOllama

    local_llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b"))

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=local_llm
    )
    print(results.to_pandas())
    return results

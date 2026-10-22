
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)
from openai import AsyncOpenAI

from src.rag import answer_question

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "evals/rag_test_cases.json"
RESULTS_DIR = ROOT_DIR / "evals/results"


def movie_id_from_chunk(chunk_id):
    return int(str(chunk_id).split(":")[0])


def retrieval_metrics(sources, expected_ids):
    retrieved = {
        movie_id_from_chunk(source["id"])
        for source in sources
    }

    expected = set(expected_ids)

    if not expected:
        return 1.0, 1.0

    coverage = len(retrieved & expected) / len(expected)

    precision = (
        len(retrieved & expected) / len(retrieved)
        if retrieved else 0.0
    )

    return coverage, precision


async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_cases = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4.1-mini",
        client=client,
        max_tokens=8192,
    )

    evaluator_embeddings = RagasOpenAIEmbeddings(
        client=client,
        model="text-embedding-3-small",
    )

    faithfulness = Faithfulness(llm=evaluator_llm)

    relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    precision = ContextPrecisionWithReference(
        llm=evaluator_llm
    )

    recall = ContextRecall(
        llm=evaluator_llm
    )

    records = []
    selected_cases = test_cases
    for index, case in enumerate(selected_cases, start=1):
        print(
            f"\n[{index}/{len(selected_cases)}] "
            f"{case['id']}: {case['question']}"
        )

        start = time.perf_counter()

        result = await asyncio.to_thread(
            answer_question,
            case["question"],
        )

        latency = time.perf_counter() - start

        sources = result["sources"]
        contexts = [
            source["text"]
            for source in sources
        ]


        scores = {}


        scores = {}

        metric_calls = {
            "faithfulness": (
                faithfulness,
                {
                    "user_input": case["question"],
                    "response": result["answer"],
                    "retrieved_contexts": contexts,
                },
            ),
            "answer_relevancy": (
                relevancy,
                {
                    "user_input": case["question"],
                    "response": result["answer"],
                },
            ),
            "context_precision": (
                precision,
                {
                    "user_input": case["question"],
                    "reference": case["reference"],
                    "retrieved_contexts": contexts,
                },
            ),
            "context_recall": (
                recall,
                {
                    "user_input": case["question"],
                    "retrieved_contexts": contexts,
                    "reference": case["reference"],
                },
            ),
        }

        for name, (metric, kwargs) in metric_calls.items():
            try:
                score = await metric.ascore(**kwargs)
                scores[name] = float(score.value)
                print(f"  {name}: {scores[name]:.3f}")
            except Exception as error:
                print(f"  {name} failed: {error}")
                scores[name] = None

        coverage, movie_precision = retrieval_metrics(
            sources,
            case["expected_movie_ids"],
        )
        record = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "reference": case["reference"],
            "answer": result["answer"],
            "expected_movie_ids": case["expected_movie_ids"],
            "retrieved_movie_ids": sorted({
                movie_id_from_chunk(source["id"])
                for source in sources
            }),
            "retrieved_sources": sources,
            "latency_seconds": round(latency, 3),
            "movie_coverage": coverage,
            "movie_precision": movie_precision,
            **scores,
        }

        records.append(record)
        checkpoint_path = RESULTS_DIR / "ragas_checkpoint.json"

        checkpoint_path.write_text(
            json.dumps(
                records,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"  Coverage: {coverage:.2f}"
            f" | Movie precision: {movie_precision:.2f}"
            f" | Latency: {latency:.2f}s"
        )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    json_path = (
        RESULTS_DIR / f"ragas_{timestamp}.json"
    )

    csv_path = (
        RESULTS_DIR / f"ragas_{timestamp}.csv"
    )

    json_path.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "movie_coverage",
        "movie_precision",
        "latency_seconds",
    ]

    print("\n=== CineScope Evaluation Summary ===\n")

    print(
        dataframe[metric_columns]
        .mean(numeric_only=True)
        .round(3)
        .to_string()
    )

    print(f"\nJSON results: {json_path}")
    print(f"CSV results: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())

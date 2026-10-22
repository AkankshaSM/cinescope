
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.rag import (
    retrieve,
    build_context,
    PROMPT,
    LLM_MODEL,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks/results"

QUESTIONS = [
    "How does The Matrix explore simulated reality?",
    "How does WALL-E explore environmental responsibility?",
    "What themes of class inequality are explored in Parasite?",
    "How does Interstellar portray time dilation and gravity?",
    "Compare how The Matrix and Inception explore reality.",
]


def benchmark_question(question):
    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    documents = retrieve(question, k=5)
    retrieval_seconds = (
        time.perf_counter() - retrieval_start
    )

    context = build_context(documents)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        max_retries=2,
    )

    chain = PROMPT | llm | StrOutputParser()

    llm_start = time.perf_counter()

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    llm_seconds = time.perf_counter() - llm_start
    total_seconds = time.perf_counter() - total_start

    return {
        "question": question,
        "retrieval_seconds": round(
            retrieval_seconds, 3
        ),
        "llm_seconds": round(llm_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "chunks_retrieved": len(documents),
        "answer_characters": len(answer),
    }


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for index, question in enumerate(QUESTIONS, 1):
        print(
            f"\n[{index}/{len(QUESTIONS)}] {question}",
            flush=True,
        )

        try:
            result = benchmark_question(question)
            records.append(result)

            print(
                f"Retrieval: "
                f"{result['retrieval_seconds']:.3f}s"
            )
            print(
                f"LLM: {result['llm_seconds']:.3f}s"
            )
            print(
                f"Total: {result['total_seconds']:.3f}s"
            )

        except Exception as error:
            print(f"FAILED: {error}")
            records.append({
                "question": question,
                "error": str(error),
            })

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output = RESULTS_DIR / f"rag_latency_{timestamp}.json"

    output.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    successful = [
        row for row in records
        if "error" not in row
    ]

    print("\n=== RAG Benchmark Summary ===")

    if successful:
        for key in (
            "retrieval_seconds",
            "llm_seconds",
            "total_seconds",
        ):
            values = [row[key] for row in successful]
            print(
                f"{key}: "
                f"{statistics.mean(values):.3f}s"
            )

    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()

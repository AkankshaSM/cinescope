
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.model_router import ModelDecision, create_llm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

RESULTS_DIR = ROOT / "benchmarks" / "results"

PROMPTS = [
    {
        "id": "M01",
        "task": "factual",
        "question": "Who directed Interstellar, and when was it released?",
    },
    {
        "id": "M02",
        "task": "research",
        "question": (
            "Explain how The Matrix explores simulated reality "
            "and human perception. Answer in 100 words or fewer."
        ),
    },
    {
        "id": "M03",
        "task": "synthesis",
        "question": (
            "Compare how WALL-E and Avatar explore environmental "
            "responsibility. Answer in 120 words or fewer."
        ),
    },
]

SYSTEM_PROMPT = """
You are CineScope, a movie research assistant.

Answer accurately and concisely.
Do not invent facts or sources.
If uncertain, acknowledge uncertainty.
Do not include unsupported citations.
"""

PROVIDERS = {
    "openai": {
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4.1-mini",
        "input_price_env": "OPENAI_INPUT_PER_MILLION",
        "output_price_env": "OPENAI_OUTPUT_PER_MILLION",
    },
    "nvidia": {
        "model_env": "NVIDIA_MODEL",
        "default_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "input_price_env": "NVIDIA_INPUT_PER_MILLION",
        "output_price_env": "NVIDIA_OUTPUT_PER_MILLION",
    },
}


def get_price(env_name):
    value = os.getenv(env_name)

    if value is None or not value.strip():
        return None

    price = float(value)

    if price < 0:
        raise ValueError(f"{env_name} cannot be negative")

    return price


def estimate_cost(provider, input_tokens, output_tokens):
    config = PROVIDERS[provider]

    input_price = get_price(config["input_price_env"])
    output_price = get_price(config["output_price_env"])

    if (
        input_price is None
        or output_price is None
        or input_tokens is None
        or output_tokens is None
    ):
        return None

    return (
        input_tokens * input_price
        + output_tokens * output_price
    ) / 1_000_000


def extract_usage(response):
    usage = response.usage_metadata or {}

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    output_details = usage.get("output_token_details") or {}
    reasoning_tokens = output_details.get("reasoning")

    if not usage:
        raw = response.response_metadata.get("token_usage", {})

        input_tokens = raw.get("prompt_tokens")
        output_tokens = raw.get("completion_tokens")
        total_tokens = raw.get("total_tokens")

        completion_details = (
            raw.get("completion_tokens_details") or {}
        )
        reasoning_tokens = completion_details.get("reasoning_tokens")

    visible_output_tokens = (
        output_tokens - reasoning_tokens
        if output_tokens is not None
        and reasoning_tokens is not None
        else None
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "non_reasoning_output_tokens": visible_output_tokens,
        "total_tokens": total_tokens,
    }


def response_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    return str(content)


def benchmark(provider, prompt, run_number):
    config = PROVIDERS[provider]

    model = os.getenv(
        config["model_env"],
        config["default_model"],
    )

    decision = ModelDecision(
        provider=provider,
        model=model,
        reason="Controlled provider benchmark",
    )

    llm = create_llm(decision)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt["question"]),
    ]

    start = time.perf_counter()

    response = llm.invoke(messages)

    latency = time.perf_counter() - start

    answer = response_text(response)
    usage = extract_usage(response)

    cost = estimate_cost(
        provider,
        usage["input_tokens"],
        usage["output_tokens"],
    )

    return {
        "run": run_number,
        "prompt_id": prompt["id"],
        "task": prompt["task"],
        "question": prompt["question"],
        "provider": provider,
        "model": model,
        "latency_seconds": round(latency, 3),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": (
            round(cost, 8)
            if cost is not None else None
        ),
        "answer": answer,
        "answer_characters": len(answer),
        "response_received": bool(answer.strip()),
        "error": None,
        "reasoning_tokens": usage["reasoning_tokens"],
        "non_reasoning_output_tokens": usage["non_reasoning_output_tokens"],
    }


def summarize(records):
    print("\n=== Provider Comparison ===\n")

    for provider in PROVIDERS:
        rows = [
            row for row in records
            if row["provider"] == provider
            and row["error"] is None
        ]

        if not rows:
            print(f"{provider}: No successful requests")
            continue

        print(f"Provider: {provider}")
        print(f"Model: {rows[0]['model']}")
        print(f"Successful requests: {len(rows)}")

        latencies = [
            row["latency_seconds"]
            for row in rows
        ]

        print(
            f"Mean latency: "
            f"{statistics.mean(latencies):.3f}s"
        )

        print(
            f"Median latency: "
            f"{statistics.median(latencies):.3f}s"
        )

        for key in ("input_tokens", "output_tokens"):
            values = [
                row[key]
                for row in rows
                if row[key] is not None
            ]

            if values:
                print(
                    f"Mean {key}: "
                    f"{statistics.mean(values):.1f}"
                )
            else:
                print(f"Mean {key}: unavailable")

        costs = [
            row["estimated_cost_usd"]
            for row in rows
            if row["estimated_cost_usd"] is not None
        ]

        if len(costs) == len(rows):
            print(
                f"Estimated cost per query: "
                f"${statistics.mean(costs):.6f}"
            )
        else:
            print("Estimated cost: not configured")

        print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    # Alternate provider order between runs to reduce
    # systematic warm-up or ordering bias.
    for run_number in range(1, args.runs + 1):
        for prompt in PROMPTS:
            order = (
                ["openai", "nvidia"]
                if run_number % 2
                else ["nvidia", "openai"]
            )

            for provider in order:
                print(
                    f"\nRun {run_number} | "
                    f"{prompt['id']} | {provider}",
                    flush=True,
                )

                try:
                    result = benchmark(
                        provider,
                        prompt,
                        run_number,
                    )

                    print(
                        f"Latency: "
                        f"{result['latency_seconds']:.3f}s"
                    )

                    print(
                        f"Input tokens: "
                        f"{result['input_tokens']}"
                    )

                    print(
                        f"Output tokens: "
                        f"{result['output_tokens']}"
                    )

                    print(
                        f"Estimated cost: "
                        f"{result['estimated_cost_usd']}"
                    )

                except Exception as error:
                    result = {
                        "run": run_number,
                        "prompt_id": prompt["id"],
                        "provider": provider,
                        "model": os.getenv(
                            PROVIDERS[provider]["model_env"],
                            PROVIDERS[provider]["default_model"],
                        ),
                        "error": str(error),
                    }

                    print(f"FAILED: {error}")

                records.append(result)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output = RESULTS_DIR / (
        f"model_comparison_{timestamp}.json"
    )

    output.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summarize(records)

    print(f"Results: {output}")


if __name__ == "__main__":
    main()

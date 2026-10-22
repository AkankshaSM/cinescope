
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks/results"

DEFAULT_URL = (
    "https://cinescope-887999373674."
    "us-west1.run.app"
)

QUESTIONS = [
    "Find science-fiction movies directed by Christopher Nolan.",
    "How does The Matrix explore simulated reality?",
    "What are the latest announcements about Dune Part Three?",
]


def benchmark_request(url, question, label):
    start = time.perf_counter()

    first_token = None
    first_event = None
    completed = False
    token_count = 0
    agents = []
    error_message = None

    with requests.post(
        f"{url.rstrip('/')}/api/chat",
        json={"message": question},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=(30, 300),
    ) as response:

        response.raise_for_status()

        event_type = "message"
        data_lines = []

        def process_event():
            nonlocal first_token
            nonlocal first_event
            nonlocal completed
            nonlocal token_count
            nonlocal agents
            nonlocal error_message

            if not data_lines:
                return

            now = time.perf_counter()

            if first_event is None:
                first_event = now - start

            payload = json.loads("\n".join(data_lines))

            if event_type == "token":
                text = payload.get("text", "")

                if text:
                    token_count += 1

                    if first_token is None:
                        first_token = now - start

            elif event_type == "trace":
                entries = payload.get("entries", [])
                agents = [
                    entry.get("node")
                    for entry in entries
                    if entry.get("node") in (
                        "research", "discovery", "news"
                    )
                ]

            elif event_type == "error":
                error_message = payload.get(
                    "message", "Unknown SSE error"
                )

            elif event_type == "done":
                completed = True

        for line in response.iter_lines(decode_unicode=True):
            if isinstance(line, bytes):
                line = line.decode("utf-8")

            if line == "":
                process_event()
                event_type = "message"
                data_lines = []
                continue

            if line.startswith("event:"):
                event_type = line[6:].strip()

            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            process_event()

    total = time.perf_counter() - start

    if error_message:
        raise RuntimeError(error_message)

    if not completed:
        raise RuntimeError(
            "SSE stream ended without a done event"
        )

    if first_token is None:
        raise RuntimeError(
            "No answer tokens received"
        )

    return {
        "label": label,
        "question": question,
        "ttft_seconds": round(first_token, 3),
        "first_event_seconds": (
            round(first_event, 3)
            if first_event is not None else None
        ),
        "total_seconds": round(total, 3),
        "stream_chunks": token_count,
        "agents": agents,
        "completed": completed,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
    )

    parser.add_argument(
        "--mode",
        choices=["cold", "warm"],
        required=True,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for index in range(args.runs):
        question = QUESTIONS[index % len(QUESTIONS)]

        print(
            f"\n[{index + 1}/{args.runs}] "
            f"{args.mode}: {question}",
            flush=True,
        )

        try:
            result = benchmark_request(
                args.url,
                question,
                args.mode,
            )

            records.append(result)

            print(
                f"TTFT: {result['ttft_seconds']:.3f}s"
            )
            print(
                f"Total: {result['total_seconds']:.3f}s"
            )
            print(f"Agents: {result['agents']}")

        except Exception as error:
            print(f"FAILED: {error}")
            records.append({
                "label": args.mode,
                "question": question,
                "error": str(error),
            })

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output = RESULTS_DIR / (
        f"cloud_{args.mode}_{timestamp}.json"
    )

    output.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    successful = [
        record for record in records
        if "error" not in record
    ]

    print("\n=== Cloud Benchmark Summary ===")

    if successful:
        for metric in (
            "ttft_seconds",
            "total_seconds",
        ):
            values = [
                record[metric]
                for record in successful
            ]

            print(
                f"{metric}: "
                f"{statistics.mean(values):.3f}s"
            )

    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()

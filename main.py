
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import logging

from src.graph import graph

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
app = FastAPI(title="CineScope")
app.mount(
    "/assets",
    StaticFiles(directory=ROOT_DIR / "static" / "assets"),
    name="assets",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def sse(event_type, data):
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


@app.get("/")
async def home():
    return FileResponse(ROOT_DIR / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest):

    async def event_stream():
        initial_state = {
            "question": request.message,
            "selected_agents": [],
            "pending_agents": [],
            "complexity": "simple",
            "research_use_web": False,
            "routing_reason": "",
            "results": [],
            "trace": [],
            "final_answer": "",
            "off_topic": False,
        }

        emitted_tokens = False
        final_answer = ""
        latest_trace = []
        sources = []
        seen_sources = set()

        try:
            async for event in graph.astream_events(
                initial_state,
                version="v2",
            ):
                event_type = event["event"]
                node = event.get("metadata", {}).get(
                    "langgraph_node"
                )

                if (
                    event_type == "on_chat_model_stream"
                    and node == "synthesis"
                ):
                    chunk = event["data"]["chunk"]
                    content = chunk.content

                    if isinstance(content, str) and content:
                        emitted_tokens = True
                        yield sse("token", {"text": content})

                if event_type == "on_chain_end":
                    if event.get("name") not in {
                        "supervisor",
                        "research",
                        "discovery",
                        "news",
                        "synthesis",
                    }:
                        continue

                    output = event.get("data", {}).get(
                        "output", {}
                    )

                    if not isinstance(output, dict):
                        continue

                    trace = output.get("trace")

                    if isinstance(trace, list):
                        latest_trace = trace
                        yield sse("trace", {
                            "entries": latest_trace
                        })

                    if node in {
                        "research",
                        "discovery",
                        "news",
                    }:
                        results = output.get("results", [])

                        if results:
                            result = results[-1]

                            for source in result.get(
                                "sources", []
                            ):
                                if not isinstance(source, dict):
                                    continue

                                url = source.get("url")

                                if url and url not in seen_sources:
                                    seen_sources.add(url)
                                    sources.append({
                                        "title": source.get(
                                            "title", url
                                        ),
                                        "url": url,
                                    })

                    if node == "synthesis":
                        final_answer = output.get(
                            "final_answer", ""
                        )

            if not emitted_tokens and final_answer:
                yield sse("token", {
                    "text": final_answer
                })

            yield sse("sources", {"items": sources})
            yield sse("done", {"status": "complete"})

        except asyncio.CancelledError:
            raise

        except Exception as error:
            logger.exception("CineScope request failed")
            yield sse("error", {
                "message": "CineScope could not complete this request."
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

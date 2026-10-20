
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

Provider = Literal["openai", "nvidia"]
Complexity = Literal["simple", "complex"]
TaskType = Literal[
    "supervisor",
    "filter_extraction",
    "discovery",
    "research",
    "news",
    "synthesis",
]


@dataclass(frozen=True)
class ModelDecision:
    provider: Provider
    model: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
        }


def select_model(
    task: TaskType,
    complexity: Complexity = "simple",
) -> ModelDecision:
    if task in ("supervisor", "filter_extraction"):
        provider = "openai"
        reason = (
            f"{task} requires reliable structured output; "
            "using the validated OpenAI integration."
        )

    elif complexity == "complex":
        provider = "openai"
        reason = (
            f"Complex {task} task requires multi-step "
            "reasoning or synthesis."
        )

    else:
        provider = "nvidia"
        reason = (
            f"Simple {task} task; using NVIDIA Nemotron "
            "for potentially lower-cost generation."
        )

    model = (
        os.environ["OPENAI_MODEL"]
        if provider == "openai"
        else os.environ["NVIDIA_MODEL"]
    )

    return ModelDecision(
        provider=provider,
        model=model,
        reason=reason,
    )


def create_llm(decision: ModelDecision) -> ChatOpenAI:
    if decision.provider == "openai":
        return ChatOpenAI(
            model=decision.model,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            max_retries=2,
        )

    return ChatOpenAI(
        model=decision.model,
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.getenv(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ),
        temperature=0,
        max_retries=2,
    )

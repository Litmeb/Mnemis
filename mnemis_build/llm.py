from __future__ import annotations

import json
from typing import Sequence, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import BuildConfig

T = TypeVar("T", bound=BaseModel)


class OpenAILLMClient:
    def __init__(self, config: BuildConfig):
        client_args = {"api_key": config.llm_api_key}
        if config.llm_base_url:
            client_args["base_url"] = config.llm_base_url
        self.client = AsyncOpenAI(**client_args)
        self.config = config

    async def complete_json(
        self,
        model: type[T],
        messages: Sequence[dict[str, str]],
        *,
        use_small_model: bool = False,
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        response = await self.client.chat.completions.create(
            model=model_name or (self.config.small_llm_model if use_small_model else self.config.llm_model),
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=list(messages),
        )
        content = response.choices[0].message.content or "{}"
        return model.model_validate(json.loads(content))

    async def complete_text(
        self,
        messages: Sequence[dict[str, str]],
        *,
        use_small_model: bool = False,
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model_name or (self.config.small_llm_model if use_small_model else self.config.llm_model),
            temperature=temperature,
            messages=list(messages),
        )
        return response.choices[0].message.content or ""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input=list(texts),
            dimensions=self.config.embedding_dim,
        )
        return [item.embedding for item in response.data]

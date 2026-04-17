from __future__ import annotations

import json
from typing import Any, Sequence, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import BuildConfig

T = TypeVar("T", bound=BaseModel)


def build_async_openai_client(*, api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
    return AsyncOpenAI(**client_args)


class OpenAILLMClient:
    def __init__(self, config: BuildConfig):
        self.client = build_async_openai_client(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
        self.config = config

    async def complete_json(
        self,
        model: type[T],
        messages: Sequence[dict[str, str]],
        *,
        use_small_model: bool = False,
        model_name: str | None = None,
        temperature: float = 0.0,
        require_json_object: bool = True,
    ) -> T:
        request: dict[str, Any] = {
            "model": model_name or (self.config.small_llm_model if use_small_model else self.config.llm_model),
            "temperature": temperature,
            "messages": list(messages),
        }
        if require_json_object:
            request["response_format"] = {"type": "json_object"}
        response = await self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or "{}"
        return self.parse_json_response(model, content)

    def parse_json_response(self, model: type[T], content: str) -> T:
        # Some prompts allow either a top-level object or a top-level array.
        return model.model_validate(self._parse_json_content(content))

    def _parse_json_content(self, content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            starts = [pos for pos in (content.find("{"), content.find("[")) if pos != -1]
            for start in sorted(starts):
                try:
                    parsed, _ = decoder.raw_decode(content[start:])
                    return parsed
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not parse JSON content: {content!r}")

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

    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        model_name: str,
        top_n: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        client = (
            self.client
            if (api_key is None or api_key == self.config.llm_api_key)
            and (base_url is None or base_url == self.config.llm_base_url)
            else build_async_openai_client(
                api_key=api_key or self.config.llm_api_key,
                base_url=base_url if base_url is not None else self.config.llm_base_url,
            )
        )
        return await client.post(
            "/rerank",
            cast_to=dict,
            body={
                "model": model_name,
                "query": query,
                "documents": list(documents),
                "top_n": top_n or len(documents),
            },
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input=list(texts),
            dimensions=self.config.embedding_dim,
        )
        return [item.embedding for item in response.data]

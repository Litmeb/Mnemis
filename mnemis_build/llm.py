from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Any, Sequence, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import BuildConfig
from .instrumentation import InstrumentationRecorder

T = TypeVar("T", bound=BaseModel)
_JSON_RETRY_LIMIT = 3
_EMBED_RETRY_LIMIT = 3
_EMBED_BATCH_SIZE = 32


def build_async_openai_client(*, api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
    return AsyncOpenAI(**client_args)


class OpenAILLMClient:
    def __init__(self, config: BuildConfig, recorder: InstrumentationRecorder | None = None):
        self.client = build_async_openai_client(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
        self.embedding_client = build_async_openai_client(
            api_key=config.embedding_api_key or config.llm_api_key,
            base_url=config.embedding_base_url or config.llm_base_url,
        )
        self.config = config
        self.recorder = recorder

    def _extract_usage(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _build_json_retry_messages(
        self,
        messages: Sequence[dict[str, str]],
        content: str,
        error: Exception,
    ) -> list[dict[str, str]]:
        snippet = content[-4000:] if len(content) > 4000 else content
        return [
            *list(messages),
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "The previous reply was not valid complete JSON for the requested schema. "
                    f"Parser error: {error}. "
                    "Return the full response again as a single complete JSON value only, with no markdown fences "
                    "and no explanatory text. If the previous response was truncated, regenerate it from scratch.\n\n"
                    f"Previous reply:\n{snippet}"
                ),
            },
        ]

    async def complete_json(
        self,
        model: type[T],
        messages: Sequence[dict[str, str]],
        *,
        stage: str | None = None,
        operation: str | None = None,
        use_small_model: bool = False,
        model_name: str | None = None,
        temperature: float = 0.0,
        require_json_object: bool = True,
    ) -> T:
        selected_model = model_name or (self.config.small_llm_model if use_small_model else self.config.llm_model)
        current_messages = list(messages)
        last_error: Exception | None = None
        last_content = "{}"
        last_finish_reason: str | None = None

        for attempt in range(1, _JSON_RETRY_LIMIT + 1):
            request: dict[str, Any] = {
                "model": selected_model,
                "temperature": temperature,
                "messages": current_messages,
            }
            if require_json_object:
                request["response_format"] = {"type": "json_object"}
            start = perf_counter()
            response = await self.client.chat.completions.create(**request)
            runtime_seconds = perf_counter() - start
            if self.recorder and stage and operation:
                usage = self._extract_usage(response)
                self.recorder.record_llm_call(
                    stage=stage,
                    operation=operation,
                    runtime_seconds=runtime_seconds,
                    model=selected_model,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    metadata={
                        "call_type": "chat.completions",
                        "response_model": model.__name__,
                        "attempt": attempt,
                    },
                )
            choice = response.choices[0]
            last_finish_reason = getattr(choice, "finish_reason", None)
            content = choice.message.content or "{}"
            last_content = content
            try:
                return self.parse_json_response(model, content)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                if attempt >= _JSON_RETRY_LIMIT:
                    break
                current_messages = self._build_json_retry_messages(messages, content, exc)

        content_snippet = last_content[:500]
        raise ValueError(
            "Could not obtain valid JSON after "
            f"{_JSON_RETRY_LIMIT} attempts for {model.__name__}. "
            f"Last finish_reason={last_finish_reason!r}. "
            f"Last error: {last_error}. "
            f"Response prefix: {content_snippet!r}"
        )

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
        stage: str | None = None,
        operation: str | None = None,
        use_small_model: bool = False,
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        selected_model = model_name or (self.config.small_llm_model if use_small_model else self.config.llm_model)
        start = perf_counter()
        response = await self.client.chat.completions.create(
            model=selected_model,
            temperature=temperature,
            messages=list(messages),
        )
        runtime_seconds = perf_counter() - start
        if self.recorder and stage and operation:
            usage = self._extract_usage(response)
            self.recorder.record_llm_call(
                stage=stage,
                operation=operation,
                runtime_seconds=runtime_seconds,
                model=selected_model,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                metadata={"call_type": "chat.completions"},
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

    def _build_embedding_request(self, texts: Sequence[str]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.embedding_model,
            "input": list(texts),
        }
        if self.config.embedding_dim > 0:
            request["dimensions"] = self.config.embedding_dim
        return request

    def _is_retryable_embedding_error(self, error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code in {408, 409, 429}:
            return True
        if isinstance(status_code, int) and status_code >= 500:
            return True
        if error.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }:
            return True
        message = str(error)
        return "No embedding data received" in message or "timed out" in message.lower()

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        attempt = 0
        last_error: Exception | None = None
        for attempt in range(1, _EMBED_RETRY_LIMIT + 1):
            start = perf_counter()
            try:
                response = await self.embedding_client.embeddings.create(**self._build_embedding_request(texts))
                runtime_seconds = perf_counter() - start
                embeddings = [item.embedding for item in getattr(response, "data", [])]
                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} embeddings, received {len(embeddings)}."
                    )
                if self.recorder:
                    usage = self._extract_usage(response)
                    self.recorder.record_llm_call(
                        stage="embeddings",
                        operation="embed",
                        runtime_seconds=runtime_seconds,
                        model=self.config.embedding_model,
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                        total_tokens=usage["total_tokens"],
                        metadata={
                            "call_type": "embeddings",
                            "input_count": len(texts),
                            "attempt": attempt,
                        },
                    )
                return embeddings
            except Exception as exc:
                last_error = exc
                if attempt >= _EMBED_RETRY_LIMIT or not self._is_retryable_embedding_error(exc):
                    break
                await asyncio.sleep(0.5 * attempt)

        raise RuntimeError(
            "Embedding request failed after "
            f"{attempt} attempt(s) for model {self.config.embedding_model!r} "
            f"with batch_size={len(texts)} and dimensions={self.config.embedding_dim}. "
            f"Last error: {last_error}"
        ) from last_error

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        items = list(texts)
        embeddings: list[list[float]] = []
        for start in range(0, len(items), _EMBED_BATCH_SIZE):
            batch = items[start : start + _EMBED_BATCH_SIZE]
            embeddings.extend(await self._embed_batch(batch))
        return embeddings

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .config import BuildConfig
from .llm import OpenAILLMClient
from .models import RerankPayload
from .prompts import RERANK_SYSTEM_PROMPT, build_rerank_user_prompt


@dataclass(slots=True)
class RerankCandidate:
    uuid: str
    text: str


@dataclass(slots=True)
class RerankBackendStatus:
    mode: str
    backend: str
    model: str | None
    requested_mode: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RerankResponse:
    scores: dict[str, float]
    status: RerankBackendStatus


class APIRerankerBackend:
    mode = "true_reranker"
    backend = "openai_compatible_rerank_api"

    def __init__(self, llm: OpenAILLMClient, config: BuildConfig):
        self.llm = llm
        self.config = config

    async def rerank(
        self,
        *,
        query: str,
        item_type: str,
        candidates: list[RerankCandidate],
    ) -> RerankResponse:
        if not self.config.reranker_model:
            raise RuntimeError("No reranker model configured.")
        if not self.config.reranker_api_key:
            raise RuntimeError("No reranker API key configured.")

        response = await self.llm.rerank(
            query=query,
            documents=[candidate.text for candidate in candidates],
            model_name=self.config.reranker_model,
            top_n=len(candidates),
            api_key=self.config.reranker_api_key,
            base_url=self.config.reranker_base_url,
        )

        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Rerank API response did not include a results list.")

        scores: dict[str, float] = {}
        for result in results:
            if not isinstance(result, dict):
                continue
            index = result.get("index")
            relevance_score = result.get("relevance_score")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue
            if relevance_score is None:
                continue
            scores[candidates[index].uuid] = float(relevance_score)

        if not scores:
            raise RuntimeError("Rerank API returned no usable relevance scores.")

        return RerankResponse(
            scores=scores,
            status=RerankBackendStatus(
                mode=self.mode,
                backend=self.backend,
                model=self.config.reranker_model,
                requested_mode=self.config.rerank_mode,
            ),
        )


class LLMScoringBackend:
    mode = "llm_scoring"
    backend = "chat_completion_json_scoring"

    def __init__(self, llm: OpenAILLMClient, config: BuildConfig):
        self.llm = llm
        self.config = config

    async def rerank(
        self,
        *,
        query: str,
        item_type: str,
        candidates: list[RerankCandidate],
    ) -> RerankResponse:
        prompt = build_rerank_user_prompt(
            query,
            item_type,
            json.dumps(
                [{"uuid": candidate.uuid, "text": candidate.text} for candidate in candidates],
                ensure_ascii=False,
                indent=2,
            ),
        )
        scores = await self.llm.complete_json(
            RerankPayload,
            [
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model_name=self.config.small_llm_model,
        )
        return RerankResponse(
            scores={item.uuid: item.score for item in scores.items},
            status=RerankBackendStatus(
                mode=self.mode,
                backend=self.backend,
                model=self.config.small_llm_model,
                requested_mode=self.config.rerank_mode,
            ),
        )


class RoutedReranker:
    def __init__(self, primary: APIRerankerBackend | None, fallback: LLMScoringBackend | None):
        self.primary = primary
        self.fallback = fallback
        self._forced_backend: APIRerankerBackend | LLMScoringBackend | None = None
        self._fallback_reason: str | None = None
        self._last_status: RerankBackendStatus | None = None

    @property
    def last_status(self) -> RerankBackendStatus | None:
        return self._last_status

    async def rerank(
        self,
        *,
        query: str,
        item_type: str,
        candidates: list[RerankCandidate],
    ) -> RerankResponse:
        backend = self._forced_backend or self.primary or self.fallback
        if backend is None:
            raise RuntimeError("No rerank backend is available.")

        try:
            response = await backend.rerank(query=query, item_type=item_type, candidates=candidates)
        except Exception as exc:
            if backend is self.primary and self.fallback is not None:
                self._fallback_reason = f"{type(exc).__name__}: {exc}"
                self._forced_backend = self.fallback
                response = await self.fallback.rerank(query=query, item_type=item_type, candidates=candidates)
                response.status.fallback_reason = self._fallback_reason
            else:
                raise
        else:
            if self._fallback_reason and backend is self.fallback:
                response.status.fallback_reason = self._fallback_reason

        self._last_status = response.status
        return response


def build_reranker(llm: OpenAILLMClient, config: BuildConfig) -> RoutedReranker:
    api_backend = APIRerankerBackend(llm, config)
    llm_backend = LLMScoringBackend(llm, config)

    if config.rerank_mode == "llm_scoring":
        return RoutedReranker(primary=None, fallback=llm_backend)

    fallback = llm_backend if config.rerank_allow_llm_fallback else None
    return RoutedReranker(primary=api_backend, fallback=fallback)

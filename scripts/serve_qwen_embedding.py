from __future__ import annotations

import argparse
import os
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency: torch") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: sentence-transformers. Run `python -m pip install -r requirements-embedding-server.txt`."
    ) from exc

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: transformers. Run `python -m pip install -r requirements-embedding-server.txt`."
    ) from exc


def _pick_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str] | list[int] | list[list[int]]
    dimensions: int | None = Field(default=None, ge=1)
    encoding_format: str | None = "float"
    user: str | None = None


class EmbeddingService:
    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.native_dimension = int(self.model.get_sentence_embedding_dimension() or 0)

    def normalize_input(self, value: str | list[str] | list[int] | list[list[int]]) -> list[str]:
        if isinstance(value, str):
            return [value]
        if not value:
            return []
        if isinstance(value[0], str):
            return [str(item) for item in value]
        if isinstance(value[0], int):
            return [self.tokenizer.decode(value, skip_special_tokens=True)]
        return [self.tokenizer.decode(item, skip_special_tokens=True) for item in value]

    def count_tokens(self, texts: list[str]) -> int:
        total = 0
        for text in texts:
            total += len(self.tokenizer.encode(text, add_special_tokens=True))
        return total

    def embed(self, texts: list[str], dimensions: int | None) -> list[list[float]]:
        if not texts:
            return []
        matrix = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        output_dim = matrix.shape[1]
        if dimensions is not None:
            if dimensions > output_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested dimensions={dimensions} exceeds model output dim={output_dim}.",
                )
            matrix = matrix[:, :dimensions]
            norms = (matrix**2).sum(axis=1, keepdims=True) ** 0.5
            matrix = matrix / norms.clip(min=1e-12)
        return matrix.tolist()


def create_app(service: EmbeddingService) -> FastAPI:
    app = FastAPI(title="Qwen Embedding Server", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": service.model_name,
            "device": service.device,
            "native_dimension": service.native_dimension,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": service.model_name,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/embeddings")
    async def create_embeddings(
        request: EmbeddingRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        # Keep OpenAI-compatible auth shape, but allow local unauthenticated calls.
        _ = authorization
        texts = service.normalize_input(request.input)
        embeddings = service.embed(texts, request.dimensions)
        prompt_tokens = service.count_tokens(texts)
        model_name = request.model or service.model_name
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": index,
                }
                for index, embedding in enumerate(embeddings)
            ],
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen3-Embedding-0.6B with an OpenAI-compatible /v1/embeddings API.")
    parser.add_argument("--host", default=os.getenv("EMBED_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("EMBED_SERVER_PORT", "8001")))
    parser.add_argument("--model", default=os.getenv("EMBED_SERVER_MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    parser.add_argument("--device", default=os.getenv("EMBED_SERVER_DEVICE"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBED_SERVER_BATCH_SIZE", "32")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _pick_device(args.device)
    service = EmbeddingService(model_name=args.model, device=device, batch_size=args.batch_size)
    app = create_app(service)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

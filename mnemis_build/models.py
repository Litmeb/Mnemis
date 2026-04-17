from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def make_uuid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class EpisodeInput(BaseModel):
    speaker: str
    content: str
    valid_at: datetime
    source_id: str
    session_id: str | None = None
    turn_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityNameExtraction(BaseModel):
    names: list[str]


class EntityRecord(BaseModel):
    uuid: str = Field(default_factory=lambda: make_uuid("entity"))
    group_id: str = ""
    name: str
    summary: str
    tag: list[str] = Field(default_factory=list)
    episode_idx: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    is_speaker: bool = False


class EdgeRecord(BaseModel):
    uuid: str = Field(default_factory=lambda: make_uuid("fact"))
    group_id: str = ""
    source_entity_name: str
    target_entity_name: str
    fact: str
    valid_at: datetime | None = None
    invalid_at: datetime | None = None


class EntityExtractionPayload(BaseModel):
    entities: list[EntityRecord]


class EdgeExtractionPayload(BaseModel):
    edges: list[EdgeRecord]

    @model_validator(mode="before")
    @classmethod
    def _coerce_top_level_shape(cls, value: object) -> object:
        if isinstance(value, list):
            return {"edges": value}
        if isinstance(value, dict):
            if "edges" in value:
                return value
            for alias in ("missing_edges", "new_edges", "additional_edges"):
                if alias in value:
                    return {"edges": value.get(alias) or []}
            for candidate in value.values():
                if (
                    isinstance(candidate, list)
                    and all(
                        isinstance(item, dict)
                        and {"source_entity_name", "target_entity_name", "fact"}.issubset(item.keys())
                        for item in candidate
                    )
                ):
                    return {"edges": candidate}
        return value


class IndexedNode(BaseModel):
    index: int
    uuid: str
    name: str
    summary: str
    tag: list[str] = Field(default_factory=list)
    layer: int = 0
    is_speaker: bool = False


class CategoryAssignment(BaseModel):
    category: str
    indexes: list[int]


class CategoryAssignmentPayload(BaseModel):
    assignments: list[CategoryAssignment]

    @model_validator(mode="before")
    @classmethod
    def _coerce_top_level_shape(cls, value: object) -> object:
        if isinstance(value, list):
            return {"assignments": value}
        return value


class CategoryDetail(BaseModel):
    name: str
    summary: str
    tag: list[str] = Field(default_factory=list)


class CategoryDetailsPayload(BaseModel):
    categories: list[CategoryDetail]


class CategoryRecord(BaseModel):
    uuid: str = Field(default_factory=lambda: make_uuid("category"))
    group_id: str
    name: str
    summary: str
    tag: list[str] = Field(default_factory=list)
    layer: int
    child_uuids: list[str]


class NodeSelection(BaseModel):
    name: str
    uuid: str
    get_all_children: bool = False


class NodeSelectionList(BaseModel):
    selections: list[NodeSelection] = Field(default_factory=list)


class RerankedItem(BaseModel):
    uuid: str
    score: float


class RerankPayload(BaseModel):
    items: list[RerankedItem] = Field(default_factory=list)

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from .base_graph import BaseGraphBuilder
from .config import BuildConfig
from .hierarchical_graph import HierarchicalGraphBuilder
from .llm import OpenAILLMClient
from .loaders import load_locomo_episodes
from .neo4j_store import Neo4jGraphStore
from .retrieval import MnemisRetriever


async def _run_rebuild_locomo(args: argparse.Namespace) -> None:
    config = BuildConfig.from_env()
    store = Neo4jGraphStore(config)
    llm = OpenAILLMClient(config)
    try:
        base_builder = BaseGraphBuilder(store, llm, config)
        hierarchy_builder = HierarchicalGraphBuilder(store, llm, config)
        episodes = load_locomo_episodes(args.data, user_index=args.user_index, group_id=args.group_id)
        await base_builder.build(args.group_id, episodes)
        await hierarchy_builder.rebuild(args.group_id)
    finally:
        await store.close()


async def _run_retrieve(args: argparse.Namespace) -> None:
    config = BuildConfig.from_env()
    store = Neo4jGraphStore(config)
    llm = OpenAILLMClient(config)
    retriever = MnemisRetriever(store, llm, config)
    try:
        payload = await retriever.retrieve(args.query, args.group_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        await store.close()


async def _run_answer(args: argparse.Namespace) -> None:
    config = BuildConfig.from_env()
    store = Neo4jGraphStore(config)
    llm = OpenAILLMClient(config)
    retriever = MnemisRetriever(store, llm, config)
    try:
        payload = await retriever.answer(args.query, args.group_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        await store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Mnemis base and hierarchical graphs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    locomo = subparsers.add_parser("rebuild-locomo", help="Ingest one LoCoMo user into local Neo4j and rebuild the hierarchy.")
    locomo.add_argument("--data", type=Path, default=Path("data/locomo.json"))
    locomo.add_argument("--user-index", type=int, default=0)
    locomo.add_argument("--group-id", required=True)
    locomo.set_defaults(func=_run_rebuild_locomo)

    retrieve = subparsers.add_parser("retrieve", help="Run System-1 + System-2 retrieval for one query.")
    retrieve.add_argument("--group-id", required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.set_defaults(func=_run_retrieve)

    answer = subparsers.add_parser("answer", help="Answer one query using the full Mnemis retrieval pipeline.")
    answer.add_argument("--group-id", required=True)
    answer.add_argument("--query", required=True)
    answer.set_defaults(func=_run_answer)
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()

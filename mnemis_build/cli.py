from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .base_graph import BaseGraphBuilder
from .config import BuildConfig
from .hierarchical_graph import HierarchicalGraphBuilder
from .instrumentation import InstrumentationRecorder
from .llm import OpenAILLMClient
from .loaders import count_locomo_users, load_locomo_episodes
from .neo4j_store import Neo4jGraphStore
from .retrieval import MnemisRetriever

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UX dependency
    tqdm = None


@dataclass(slots=True)
class _UserBuildResult:
    user_index: int
    group_id: str
    instrumentation_reports: dict[str, str]


class _UserBuildProgressReporter:
    def __init__(self, *, total_users: int, concurrency: int):
        self.total_users = total_users
        self.concurrency = max(1, min(concurrency, total_users or 1))
        self._lock = asyncio.Lock()
        self._slots: asyncio.Queue[int] = asyncio.Queue()
        for slot in range(1, self.concurrency + 1):
            self._slots.put_nowait(slot)
        self._slot_bars: dict[int, object] = {}
        self._active_slots: set[int] = set()
        self._closed = False
        self._supports_tqdm = tqdm is not None and total_users > 0
        self._total_bar = (
            tqdm(
                total=total_users,
                desc="LoCoMo users",
                unit="user",
                position=0,
                dynamic_ncols=True,
            )
            if self._supports_tqdm
            else None
        )

    async def start_user(self, user_index: int, group_id: str, total_turns: int) -> int:
        slot = await self._slots.get()
        async with self._lock:
            self._active_slots.add(slot)
            if self._supports_tqdm:
                self._slot_bars[slot] = tqdm(
                    total=max(1, total_turns),
                    desc=f"user {user_index} base graph",
                    unit="turn",
                    position=slot,
                    leave=False,
                    dynamic_ncols=True,
                )
                self._slot_bars[slot].set_postfix_str(f"{group_id} 0/{total_turns}")
            else:
                print(f"[build] user {user_index} started ({group_id}) 0/{total_turns} turns")
        return slot

    async def advance_turns(
        self,
        slot: int,
        *,
        user_index: int,
        group_id: str,
        completed_turns: int,
        total_turns: int,
    ) -> None:
        async with self._lock:
            if self._supports_tqdm:
                slot_bar = self._slot_bars.get(slot)
                if slot_bar is None:
                    return
                delta = completed_turns - slot_bar.n
                if delta > 0:
                    slot_bar.update(delta)
                slot_bar.set_description_str(f"user {user_index} base graph")
                slot_bar.set_postfix_str(f"{group_id} {completed_turns}/{total_turns}")
            else:
                print(f"[build] user {user_index} ({group_id}) {completed_turns}/{total_turns} turns")

    async def mark_hierarchy(self, slot: int, *, user_index: int, group_id: str, total_turns: int) -> None:
        async with self._lock:
            if self._supports_tqdm:
                slot_bar = self._slot_bars.get(slot)
                if slot_bar is None:
                    return
                if slot_bar.n < total_turns:
                    slot_bar.update(total_turns - slot_bar.n)
                slot_bar.set_description_str(f"user {user_index} hierarchy")
                slot_bar.set_postfix_str(f"{group_id} {total_turns}/{total_turns}")
            else:
                print(f"[build] user {user_index} hierarchy ({group_id}) {total_turns}/{total_turns} turns")

    async def finish_user(self, slot: int, *, user_index: int, group_id: str) -> None:
        async with self._lock:
            was_active = slot in self._active_slots
            self._active_slots.discard(slot)
            if self._supports_tqdm:
                slot_bar = self._slot_bars.pop(slot, None)
                if slot_bar is not None:
                    slot_bar.close()
                    if self._total_bar is not None:
                        self._total_bar.update(1)
                        self._total_bar.set_postfix_str(f"last=user {user_index}")
                    tqdm.write(f"[build] user {user_index} finished ({group_id})")
            elif was_active:
                print(f"[build] user {user_index} finished ({group_id})")
        if was_active and not self._closed:
            self._slots.put_nowait(slot)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._active_slots.clear()
            for slot_bar in self._slot_bars.values():
                slot_bar.close()
            self._slot_bars.clear()
            if self._total_bar is not None:
                self._total_bar.close()


async def _rebuild_locomo_user(
    *,
    config: BuildConfig,
    user_index: int,
    group_id: str,
    episodes: list = None,
    progress_callback=None,
    hierarchy_callback=None,
) -> _UserBuildResult:
    store = Neo4jGraphStore(config)
    recorder = InstrumentationRecorder(run_name=f"rebuild_locomo_{group_id}")
    llm = OpenAILLMClient(config, recorder=recorder)
    try:
        base_builder = BaseGraphBuilder(store, llm, config)
        hierarchy_builder = HierarchicalGraphBuilder(store, llm, config)
        if episodes is None:
            raise RuntimeError("episodes must be provided for LoCoMo rebuilds.")
        await store.clear_group(group_id)
        with recorder.stage_timer(
            "base_graph_ingestion",
            "build",
            metadata={"episode_count": len(episodes), "group_id": group_id, "user_index": user_index},
        ):
            await base_builder.build(group_id, episodes, progress_callback=progress_callback)
        if hierarchy_callback is not None:
            await hierarchy_callback(len(episodes))
        with recorder.stage_timer(
            "hierarchical_graph_ingestion",
            "rebuild",
            metadata={"group_id": group_id, "user_index": user_index},
        ):
            await hierarchy_builder.rebuild(group_id)
        report_dir = Path(os.getenv("MNEMIS_INSTRUMENTATION_DIR", "results/instrumentation"))
        report_paths = recorder.write_reports(report_dir, stem=f"rebuild_locomo_{group_id}")
        return _UserBuildResult(
            user_index=user_index,
            group_id=group_id,
            instrumentation_reports=report_paths,
        )
    finally:
        await store.close()


async def _run_rebuild_locomo(args: argparse.Namespace) -> None:
    config = BuildConfig.from_env()
    episodes = load_locomo_episodes(args.data, user_index=args.user_index, group_id=args.group_id)
    result = await _rebuild_locomo_user(
        config=config,
        user_index=args.user_index,
        group_id=args.group_id,
        episodes=episodes,
    )
    print(json.dumps({"instrumentation_reports": result.instrumentation_reports}, ensure_ascii=False, indent=2))


async def _run_rebuild_locomo_all(args: argparse.Namespace) -> None:
    config = BuildConfig.from_env()
    total_users = count_locomo_users(args.data)
    user_indexes = list(range(total_users))
    concurrency = max(1, min(args.max_concurrent_users, len(user_indexes) or 1))
    progress = _UserBuildProgressReporter(total_users=len(user_indexes), concurrency=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def rebuild_one(user_index: int) -> _UserBuildResult:
        async with semaphore:
            group_id = f"{args.group_id_prefix}_{user_index}"
            episodes = load_locomo_episodes(args.data, user_index=user_index, group_id=group_id)
            slot = await progress.start_user(user_index, group_id, len(episodes))
            try:
                result = await _rebuild_locomo_user(
                    config=config,
                    user_index=user_index,
                    group_id=group_id,
                    episodes=episodes,
                    progress_callback=lambda completed_turns, total_turns, episode: progress.advance_turns(
                        slot,
                        user_index=user_index,
                        group_id=group_id,
                        completed_turns=completed_turns,
                        total_turns=total_turns,
                    ),
                    hierarchy_callback=lambda total_turns: progress.mark_hierarchy(
                        slot,
                        user_index=user_index,
                        group_id=group_id,
                        total_turns=total_turns,
                    ),
                )
                return result
            finally:
                await progress.finish_user(slot, user_index=user_index, group_id=group_id)

    try:
        raw_results = await asyncio.gather(
            *(rebuild_one(user_index) for user_index in user_indexes),
            return_exceptions=True,
        )
    finally:
        await progress.close()

    results: list[_UserBuildResult] = []
    failures: list[tuple[int, BaseException]] = []
    for user_index, raw_result in zip(user_indexes, raw_results):
        if isinstance(raw_result, BaseException):
            failures.append((user_index, raw_result))
        else:
            results.append(raw_result)

    if failures:
        lines = [
            f"user {user_index}: {type(error).__name__}: {error}"
            for user_index, error in failures[:10]
        ]
        if len(failures) > 10:
            lines.append(f"... and {len(failures) - 10} more failures")
        raise RuntimeError(
            "rebuild-locomo-all failed for "
            f"{len(failures)}/{len(user_indexes)} users.\n" + "\n".join(lines)
        )

    payload = {
        "user_count": len(results),
        "group_id_prefix": args.group_id_prefix,
        "results": [
            {
                "user_index": result.user_index,
                "group_id": result.group_id,
                "instrumentation_reports": result.instrumentation_reports,
            }
            for result in sorted(results, key=lambda item: item.user_index)
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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

    locomo_all = subparsers.add_parser(
        "rebuild-locomo-all",
        help="Ingest all LoCoMo users into local Neo4j with user-level concurrency.",
    )
    locomo_all.add_argument("--data", type=Path, default=Path("data/locomo.json"))
    locomo_all.add_argument("--group-id-prefix", default="locomo_user")
    locomo_all.add_argument("--max-concurrent-users", type=int, default=4)
    locomo_all.set_defaults(func=_run_rebuild_locomo_all)

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

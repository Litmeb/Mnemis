from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .base_graph import BaseGraphBuilder
from .config import BuildConfig
from .hierarchical_graph import HierarchicalGraphBuilder
from .instrumentation import InstrumentationRecorder
from .llm import OpenAILLMClient
from .loaders import count_locomo_users, load_locomo_episodes
from .logging_utils import configure_logging, get_logger
from .neo4j_store import Neo4jGraphStore
from .retrieval import MnemisRetriever
from .timing import log_timed_step

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UX dependency
    tqdm = None


@dataclass(slots=True)
class _UserBuildResult:
    user_index: int
    group_id: str
    instrumentation_reports: dict[str, str]


_TURN_DONE_PATTERN = re.compile(
    r"turn done \| group_id=(?P<group_id>[^,]+), turn=(?P<turn>\d+)/(?P<total>\d+), source_id=(?P<source_id>[^,]+)"
)


def _parse_user_index_list(raw_value: str) -> list[int]:
    indexes: list[int] = []
    seen: set[int] = set()
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            raise argparse.ArgumentTypeError("user indexes must be a comma-separated list like 1,3,7")
        try:
            index = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid user index {item!r}; expected integers like 1,3,7") from exc
        if index < 0:
            raise argparse.ArgumentTypeError("user indexes must be non-negative")
        if index in seen:
            continue
        seen.add(index)
        indexes.append(index)
    if not indexes:
        raise argparse.ArgumentTypeError("at least one user index is required")
    return indexes


def _resolve_user_indexes(selected_indexes: list[int] | None, *, total_users: int) -> list[int]:
    if not selected_indexes:
        return list(range(total_users))
    invalid_indexes = [index for index in selected_indexes if index >= total_users]
    if invalid_indexes:
        invalid_text = ", ".join(str(index) for index in invalid_indexes)
        raise RuntimeError(
            f"Requested user indexes out of range for dataset with {total_users} users: {invalid_text}"
        )
    return selected_indexes


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
            get_logger("cli.progress").info(
                "user slot acquired | user_index=%s, group_id=%s, slot=%s, total_turns=%s",
                user_index,
                group_id,
                slot,
                total_turns,
            )
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
            get_logger("cli.progress").info(
                "turn progress | user_index=%s, group_id=%s, slot=%s, completed_turns=%s, total_turns=%s",
                user_index,
                group_id,
                slot,
                completed_turns,
                total_turns,
            )
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
            get_logger("cli.progress").info(
                "hierarchy phase | user_index=%s, group_id=%s, slot=%s, total_turns=%s",
                user_index,
                group_id,
                slot,
                total_turns,
            )
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
            get_logger("cli.progress").info(
                "user slot released | user_index=%s, group_id=%s, slot=%s, was_active=%s",
                user_index,
                group_id,
                slot,
                was_active,
            )
            if self._supports_tqdm:
                slot_bar = self._slot_bars.pop(slot, None)
                if slot_bar is not None:
                    slot_bar.close()
                    if self._total_bar is not None:
                        self._total_bar.update(1)
                        self._total_bar.set_postfix_str(f"last=user {user_index}")
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


def _find_latest_log_path() -> Path | None:
    base_dir = Path(os.getenv("MNEMIS_LOG_DIR", "results/logs"))
    if not base_dir.exists():
        return None
    candidates = sorted(base_dir.glob("*/mnemis_build.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_resume_log_progress(log_path: Path | None) -> dict[str, str]:
    if log_path is None or not log_path.exists():
        return {}
    progress: dict[str, str] = {}
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            match = _TURN_DONE_PATTERN.search(line)
            if match is None:
                continue
            progress[match.group("group_id")] = match.group("source_id")
    except OSError:
        return {}
    return progress


def _compute_resume_start_index(
    episodes: list,
    completed_source_ids: set[str],
    *,
    resume_completed_source_id: str | None = None,
) -> int:
    resume_index = 0
    for episode in episodes:
        if episode.source_id not in completed_source_ids:
            break
        resume_index += 1
    if resume_index == 0 and resume_completed_source_id:
        for index, episode in enumerate(episodes):
            if episode.source_id == resume_completed_source_id:
                return index + 1
    return resume_index


def _resolve_resume_start_index(
    episodes: list,
    completed_source_ids: set[str],
    *,
    group_id: str,
    resume_completed_source_id: str | None = None,
) -> int:
    start_index = _compute_resume_start_index(
        episodes,
        completed_source_ids,
        resume_completed_source_id=resume_completed_source_id,
    )
    has_resume_evidence = bool(completed_source_ids or resume_completed_source_id)
    if not has_resume_evidence:
        raise RuntimeError(
            f"Resume requested for {group_id!r}, but no completed-turn markers were found in Neo4j "
            "and no matching completed turn was recovered from the resume log. "
            "Aborting to avoid replaying from turn 0 over existing data."
        )
    if start_index <= 0:
        raise RuntimeError(
            f"Resume requested for {group_id!r}, but the recovered progress could not be mapped onto the current "
            "episode list. Aborting to avoid replaying from turn 0 over existing data."
        )
    return start_index


async def _rebuild_locomo_user(
    *,
    config: BuildConfig,
    user_index: int,
    group_id: str,
    episodes: list = None,
    progress_callback=None,
    hierarchy_callback=None,
    resume: bool = False,
    resume_completed_source_id: str | None = None,
    reuse_existing_base: bool = False,
) -> _UserBuildResult:
    logger = get_logger("cli.rebuild_user")
    store = Neo4jGraphStore(config)
    recorder = InstrumentationRecorder(run_name=f"rebuild_locomo_{group_id}")
    llm = OpenAILLMClient(config, recorder=recorder)
    try:
        base_builder = BaseGraphBuilder(store, llm, config)
        hierarchy_builder = HierarchicalGraphBuilder(store, llm, config)
        if episodes is None:
            raise RuntimeError("episodes must be provided for LoCoMo rebuilds.")
        logger.info(
            "rebuild user start | user_index=%s, group_id=%s, episode_count=%s, reuse_existing_base=%s",
            user_index,
            group_id,
            len(episodes),
            reuse_existing_base,
        )
        start_index = 0
        if reuse_existing_base:
            logger.info(
                "reuse existing base graph | user_index=%s, group_id=%s, skip_clear_group=%s, skip_base_graph_build=%s",
                user_index,
                group_id,
                True,
                True,
            )
        elif resume:
            completed_source_ids = set(await store.fetch_completed_episode_source_ids(group_id))
            if resume_completed_source_id:
                completed_source_ids.add(resume_completed_source_id)
            start_index = _resolve_resume_start_index(
                episodes,
                completed_source_ids,
                group_id=group_id,
                resume_completed_source_id=resume_completed_source_id,
            )
            logger.info(
                "resume state | user_index=%s, group_id=%s, completed_turns=%s, total_turns=%s, resume_completed_source_id=%s",
                user_index,
                group_id,
                start_index,
                len(episodes),
                resume_completed_source_id,
            )
            if progress_callback is not None and start_index > 0:
                await progress_callback(start_index, len(episodes), episodes[start_index - 1])
        else:
            with log_timed_step("clear_group", logger_name="cli.rebuild_user", group_id=group_id):
                await store.clear_group(group_id)
        if reuse_existing_base:
            logger.info(
                "skip base_graph.build | user_index=%s, group_id=%s, reason=reuse_existing_base",
                user_index,
                group_id,
            )
        else:
            with recorder.stage_timer(
                "base_graph_ingestion",
                "build",
                metadata={
                    "episode_count": len(episodes),
                    "group_id": group_id,
                    "user_index": user_index,
                    "resume": resume,
                    "start_index": start_index,
                    "reuse_existing_base": reuse_existing_base,
                },
            ):
                with log_timed_step(
                    "base_graph.build",
                    logger_name="cli.rebuild_user",
                    user_index=user_index,
                    group_id=group_id,
                    episode_count=len(episodes),
                    resume=resume,
                    start_index=start_index,
                ):
                    await base_builder.build(
                        group_id,
                        episodes,
                        progress_callback=progress_callback,
                        start_index=start_index,
                    )
        if hierarchy_callback is not None:
            await hierarchy_callback(len(episodes))
        with recorder.stage_timer(
            "hierarchical_graph_ingestion",
            "rebuild",
            metadata={"group_id": group_id, "user_index": user_index},
        ):
            with log_timed_step(
                "hierarchy.rebuild",
                logger_name="cli.rebuild_user",
                user_index=user_index,
                group_id=group_id,
            ):
                await hierarchy_builder.rebuild(group_id)
        report_dir = Path(os.getenv("MNEMIS_INSTRUMENTATION_DIR", "results/instrumentation"))
        report_paths = recorder.write_reports(report_dir, stem=f"rebuild_locomo_{group_id}")
        logger.info(
            "rebuild user done | user_index=%s, group_id=%s, reports=%s",
            user_index,
            group_id,
            report_paths,
        )
        return _UserBuildResult(
            user_index=user_index,
            group_id=group_id,
            instrumentation_reports=report_paths,
        )
    finally:
        logger.info("closing neo4j store | user_index=%s, group_id=%s", user_index, group_id)
        await store.close()


async def _run_rebuild_locomo(args: argparse.Namespace) -> None:
    get_logger("cli").info("command rebuild-locomo start | args=%s", vars(args))
    config = BuildConfig.from_env()
    episodes = load_locomo_episodes(args.data, user_index=args.user_index, group_id=args.group_id)
    resume_log_path = Path(args.resume_log) if args.resume_log else (_find_latest_log_path() if args.resume else None)
    resume_progress = _load_resume_log_progress(resume_log_path) if args.resume else {}
    result = await _rebuild_locomo_user(
        config=config,
        user_index=args.user_index,
        group_id=args.group_id,
        episodes=episodes,
        resume=args.resume,
        resume_completed_source_id=resume_progress.get(args.group_id),
        reuse_existing_base=args.reuse_existing_base,
    )
    print(json.dumps({"instrumentation_reports": result.instrumentation_reports}, ensure_ascii=False, indent=2))
    get_logger("cli").info("command rebuild-locomo done | group_id=%s", args.group_id)


async def _run_rebuild_locomo_all(args: argparse.Namespace) -> None:
    logger = get_logger("cli")
    logger.info("command rebuild-locomo-all start | args=%s", vars(args))
    config = BuildConfig.from_env()
    resume_log_path = Path(args.resume_log) if args.resume_log else (_find_latest_log_path() if args.resume else None)
    resume_progress = _load_resume_log_progress(resume_log_path) if args.resume else {}
    if args.resume:
        logger.info("resume log selected | path=%s, groups=%s", resume_log_path, len(resume_progress))
    total_users = count_locomo_users(args.data)
    user_indexes = _resolve_user_indexes(args.user_indexes, total_users=total_users)
    logger.info(
        "rebuild user selection | requested=%s, selected=%s, dataset_user_count=%s",
        args.user_indexes,
        user_indexes,
        total_users,
    )
    concurrency = max(1, min(args.max_concurrent_users, len(user_indexes) or 1))
    progress = _UserBuildProgressReporter(total_users=len(user_indexes), concurrency=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def rebuild_one(user_index: int) -> _UserBuildResult:
        async with semaphore:
            group_id = f"{args.group_id_prefix}_{user_index}"
            logger.info("user rebuild task start | user_index=%s, group_id=%s", user_index, group_id)
            episodes = load_locomo_episodes(args.data, user_index=user_index, group_id=group_id)
            slot = await progress.start_user(user_index, group_id, len(episodes))
            try:
                result = await _rebuild_locomo_user(
                    config=config,
                    user_index=user_index,
                    group_id=group_id,
                    episodes=episodes,
                    resume=args.resume,
                    resume_completed_source_id=resume_progress.get(group_id),
                    reuse_existing_base=args.reuse_existing_base,
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
                logger.info("user rebuild task done | user_index=%s, group_id=%s", user_index, group_id)
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
        logger.error("command rebuild-locomo-all failed | failure_count=%s", len(failures))
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
    logger.info("command rebuild-locomo-all done | user_count=%s", len(results))


async def _run_retrieve(args: argparse.Namespace) -> None:
    get_logger("cli").info("command retrieve start | args=%s", vars(args))
    config = BuildConfig.from_env()
    store = Neo4jGraphStore(config)
    llm = OpenAILLMClient(config)
    retriever = MnemisRetriever(store, llm, config)
    try:
        payload = await retriever.retrieve(args.query, args.group_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        get_logger("cli").info("command retrieve done | group_id=%s", args.group_id)
        await store.close()


async def _run_answer(args: argparse.Namespace) -> None:
    get_logger("cli").info("command answer start | args=%s", vars(args))
    config = BuildConfig.from_env()
    store = Neo4jGraphStore(config)
    llm = OpenAILLMClient(config)
    retriever = MnemisRetriever(store, llm, config)
    try:
        payload = await retriever.answer(args.query, args.group_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        get_logger("cli").info("command answer done | group_id=%s", args.group_id)
        await store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Mnemis base and hierarchical graphs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    locomo = subparsers.add_parser("rebuild-locomo", help="Ingest one LoCoMo user into local Neo4j and rebuild the hierarchy.")
    locomo.add_argument("--data", type=Path, default=Path("data/locomo.json"))
    locomo.add_argument("--user-index", type=int, default=0)
    locomo.add_argument("--group-id", required=True)
    locomo.add_argument("--resume", action="store_true", help="Resume from completed turns instead of clearing the group.")
    locomo.add_argument("--resume-log", type=Path, help="Optional build log to mine for the last completed turn when DB markers are missing.")
    locomo.add_argument(
        "--reuse-existing-base",
        action="store_true",
        help="Reuse the existing base graph for this group, skip clear_group and skip base_graph.build.",
    )
    locomo.set_defaults(func=_run_rebuild_locomo)

    locomo_all = subparsers.add_parser(
        "rebuild-locomo-all",
        help="Ingest all LoCoMo users into local Neo4j with user-level concurrency.",
    )
    locomo_all.add_argument("--data", type=Path, default=Path("data/locomo.json"))
    locomo_all.add_argument("--group-id-prefix", default="locomo_user")
    locomo_all.add_argument(
        "--user-index",
        dest="user_indexes",
        type=_parse_user_index_list,
        help="Optional comma-separated list of user indexes to rebuild, for example 1,3,7.",
    )
    locomo_all.add_argument("--max-concurrent-users", type=int, default=4)
    locomo_all.add_argument("--resume", action="store_true", help="Resume each user from completed turns instead of clearing the group.")
    locomo_all.add_argument("--resume-log", type=Path, help="Optional build log to mine for the last completed turn when DB markers are missing.")
    locomo_all.add_argument(
        "--reuse-existing-base",
        action="store_true",
        help="Reuse each group's existing base graph, skip clear_group and skip base_graph.build.",
    )
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
    log_path = configure_logging()
    get_logger("cli").info("build_mnemis_graph main start | log_path=%s", log_path)
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()

import asyncio

import mnemis_build.cli as cli_module
from mnemis_build.cli import _UserBuildProgressReporter


class _FakeBar:
    def __init__(self, total=0, desc="", unit="", position=0, leave=True, dynamic_ncols=False):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.position = position
        self.leave = leave
        self.dynamic_ncols = dynamic_ncols
        self.n = 0
        self.closed = False
        self.postfix = ""

    def update(self, delta: int) -> None:
        self.n += delta

    def set_postfix_str(self, value: str) -> None:
        self.postfix = value

    def set_description_str(self, value: str) -> None:
        self.desc = value

    def close(self) -> None:
        self.closed = True


class _FakeTqdm:
    def __init__(self) -> None:
        self.instances: list[_FakeBar] = []
        self.writes: list[str] = []

    def __call__(self, **kwargs) -> _FakeBar:
        bar = _FakeBar(**kwargs)
        self.instances.append(bar)
        return bar

    def write(self, message: str) -> None:
        self.writes.append(message)


def test_progress_reporter_finish_after_close_is_safe(monkeypatch) -> None:
    fake_tqdm = _FakeTqdm()
    monkeypatch.setattr(cli_module, "tqdm", fake_tqdm)
    reporter = _UserBuildProgressReporter(total_users=1, concurrency=1)

    async def _exercise_reporter() -> None:
        slot = await reporter.start_user(0, "locomo_user_0", 2)
        await reporter.close()
        await reporter.finish_user(slot, user_index=0, group_id="locomo_user_0")

    asyncio.run(_exercise_reporter())

    assert len(fake_tqdm.instances) == 2
    assert all(bar.closed for bar in fake_tqdm.instances)


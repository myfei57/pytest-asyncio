"""
Integration tests for the completion-event ordering barrier.

The tests run pytest in isolated pytester projects and inspect the barrier
registry through a snapshot written at ``pytest_sessionfinish``.
"""

from __future__ import annotations

import json
from textwrap import dedent

from pytest import Pytester

# Dumps the internal completion registry to JSON so that the outer test can
# assert on the exact publication order of the lifecycle events.
_SNAPSHOT_CONFTEST = dedent("""\
    import json


    def pytest_sessionfinish(session):
        from pytest_asyncio.plugin import _get_completion_registry

        registry = _get_completion_registry(session.config)
        with open("completion_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(registry.snapshot(), f)
    """)


def _load_snapshot(pytester: Pytester) -> dict:
    path = pytester.path / "completion_snapshot.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _phases(snapshot: dict, test_id: str) -> list[str]:
    return [event["phase"] for event in snapshot["events"][test_id]]


def _seqs(snapshot: dict, test_id: str) -> list[int]:
    return [event["seq"] for event in snapshot["events"][test_id]]


_EXPECTED_PHASES = ["TEST", "FIXTURE_TEARDOWN", "LOOP_CLOSE"]

# A globally installed third-party plugin (pytest-tornasync) uses hook
# configuration options that pytest 10 removed. Blocking it keeps the
# pytester subprocesses independent of the host environment; the option is
# a no-op when the plugin is not installed.
_RUN_ARGS = ("--asyncio-mode=strict", "-p", "no:tornado")


def test_lifecycle_events_are_published_in_phase_order(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(_SNAPSHOT_CONFTEST)
    pytester.makepyfile(
        dedent("""\
        import pytest
        import pytest_asyncio


        @pytest_asyncio.fixture
        async def async_fixture():
            yield


        @pytest.mark.asyncio
        async def test_one(async_fixture):
            pass
        """)
    )
    result = pytester.runpytest(*_RUN_ARGS)
    result.assert_outcomes(passed=1)

    snapshot = _load_snapshot(pytester)
    assert snapshot["errors"] == []
    test_id = next(iter(snapshot["events"]))
    assert _phases(snapshot, test_id) == _EXPECTED_PHASES
    assert _seqs(snapshot, test_id) == [0, 1, 2]
    assert snapshot["closed"][test_id] is True
    assert snapshot["missing"][test_id] == []


def test_failing_test_keeps_its_failure_and_the_event_order(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(_SNAPSHOT_CONFTEST)
    pytester.makepyfile(
        dedent("""\
        import pytest


        @pytest.mark.asyncio
        async def test_failure():
            assert False
        """)
    )
    result = pytester.runpytest(*_RUN_ARGS)
    result.assert_outcomes(failed=1)

    snapshot = _load_snapshot(pytester)
    assert snapshot["errors"] == []
    test_id = next(iter(snapshot["events"]))
    assert _phases(snapshot, test_id) == _EXPECTED_PHASES
    assert snapshot["missing"][test_id] == []


def test_teardown_failure_keeps_result_and_event_order(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(_SNAPSHOT_CONFTEST)
    pytester.makepyfile(
        dedent("""\
        import pytest
        import pytest_asyncio


        @pytest_asyncio.fixture
        async def broken_teardown():
            yield
            raise RuntimeError("boom in teardown")


        @pytest.mark.asyncio
        async def test_uses_broken_fixture(broken_teardown):
            pass
        """)
    )
    result = pytester.runpytest(*_RUN_ARGS)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*RuntimeError: boom in teardown*"])

    snapshot = _load_snapshot(pytester)
    assert snapshot["errors"] == []
    test_id = next(iter(snapshot["events"]))
    assert _phases(snapshot, test_id) == _EXPECTED_PHASES
    assert snapshot["closed"][test_id] is True
    assert snapshot["missing"][test_id] == []


def test_events_are_isolated_between_tests(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(_SNAPSHOT_CONFTEST)
    pytester.makepyfile(
        dedent("""\
        import pytest


        @pytest.mark.asyncio
        async def test_first():
            pass


        @pytest.mark.asyncio
        async def test_second():
            pass
        """)
    )
    result = pytester.runpytest(*_RUN_ARGS)
    result.assert_outcomes(passed=2)

    snapshot = _load_snapshot(pytester)
    assert snapshot["errors"] == []
    assert len(snapshot["events"]) == 2
    for test_id in snapshot["events"]:
        # Each barrier numbers its own events from zero, proving that no
        # event from the other test leaked across the boundary.
        assert _seqs(snapshot, test_id) == [0, 1, 2]
        assert _phases(snapshot, test_id) == _EXPECTED_PHASES


def test_module_scoped_runner_defers_loop_close_to_scope_teardown(
    pytester: Pytester,
):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = module")
    pytester.makeconftest(_SNAPSHOT_CONFTEST)
    pytester.makepyfile(
        dedent("""\
        import pytest
        import pytest_asyncio


        @pytest_asyncio.fixture(scope="module", loop_scope="module")
        async def module_fixture():
            yield


        pytestmark = pytest.mark.asyncio(loop_scope="module")


        async def test_first(module_fixture):
            pass


        async def test_second(module_fixture):
            pass
        """)
    )
    result = pytester.runpytest(*_RUN_ARGS)
    result.assert_outcomes(passed=2)

    snapshot = _load_snapshot(pytester)
    assert snapshot["errors"] == []
    assert len(snapshot["events"]) == 2
    assert all(snapshot["closed"].values())
    for test_id in snapshot["events"]:
        assert _phases(snapshot, test_id) == _EXPECTED_PHASES
        assert snapshot["missing"][test_id] == []


def test_skipped_async_test_does_not_leave_an_open_barrier(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(_SNAPSHOT_CONFTEST)
    pytester.makepyfile(
        dedent("""\
        import pytest


        @pytest.mark.asyncio
        @pytest.mark.skip
        async def test_skipped():
            pass
        """)
    )
    result = pytester.runpytest(*_RUN_ARGS)
    result.assert_outcomes(skipped=1)

    snapshot = _load_snapshot(pytester)
    assert snapshot["errors"] == []
    test_id = next(iter(snapshot["events"]))
    assert snapshot["closed"][test_id] is True
    # The call and the loop never ran; the teardown milestone still did.
    assert snapshot["missing"][test_id] == [0, 2]
    assert _phases(snapshot, test_id) == ["FIXTURE_TEARDOWN"]


def test_concurrent_tasks_do_not_perturb_the_event_order(pytester: Pytester):
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(_SNAPSHOT_CONFTEST)
    pytester.makepyfile(
        dedent("""\
        import asyncio

        import pytest


        @pytest.mark.asyncio
        async def test_concurrent_tasks():
            async def worker(index):
                await asyncio.sleep(0.01 * (3 - index))
                return index

            results = await asyncio.gather(
                *(worker(index) for index in range(3))
            )
            assert sorted(results) == [0, 1, 2]
        """)
    )
    result = pytester.runpytest(*_RUN_ARGS)
    result.assert_outcomes(passed=1)

    snapshot = _load_snapshot(pytester)
    assert snapshot["errors"] == []
    test_id = next(iter(snapshot["events"]))
    assert _phases(snapshot, test_id) == _EXPECTED_PHASES
    assert snapshot["missing"][test_id] == []

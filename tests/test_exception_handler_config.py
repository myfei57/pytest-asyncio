from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator
from textwrap import dedent
from typing import Any

import pytest
from pytest import Pytester

from pytest_asyncio.exception_config import (
    UNSET,
    ExceptionConfigError,
    ExceptionConfigRestoreError,
    ExceptionHandlerConfigMerger,
    ExceptionHandlerSettings,
    LoopConfigSnapshot,
    MergedExceptionHandlerConfig,
    ScopedSettings,
    UnknownExceptionHandlerError,
    apply_loop_config,
    loop_exception_config,
    restore_loop_config,
    snapshot_loop_config,
)


def _handler_one(loop: asyncio.AbstractEventLoop, context: object) -> None:
    raise NotImplementedError


def _handler_two(loop: asyncio.AbstractEventLoop, context: object) -> None:
    raise NotImplementedError


def _layer(scope: str, nodeid: str, **values: Any) -> ScopedSettings:
    return ScopedSettings(scope, nodeid, ExceptionHandlerSettings(**values))


@pytest.fixture
def fresh_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Merger unit tests
# ---------------------------------------------------------------------------


def test_merge_sources_and_shadowing_chains() -> None:
    merger = ExceptionHandlerConfigMerger(
        [
            _layer(
                "session",
                "session",
                debug=True,
                slow_callback_duration=1.0,
            ),
            _layer(
                "module",
                "test_mod.py",
                debug=False,
                exception_handler=_handler_one,
            ),
            _layer(
                "function",
                "test_mod.py::test_x",
                exception_handler=_handler_two,
                slow_callback_duration=2.5,
            ),
        ]
    )

    merged = merger.merge()

    assert merged.exception_handler is _handler_two
    assert merged.debug is False
    assert merged.slow_callback_duration == 2.5

    handler_source = merged.sources["exception_handler"]
    debug_source = merged.sources["debug"]
    slow_source = merged.sources["slow_callback_duration"]
    assert handler_source is not None
    assert debug_source is not None
    assert slow_source is not None
    assert handler_source.scope == "function"
    assert handler_source.nodeid == "test_mod.py::test_x"
    assert debug_source.scope == "module"
    assert slow_source.scope == "function"

    handler_chain = merged.shadow_chains["exception_handler"]
    assert [entry.scope for entry in handler_chain] == ["module", "function"]
    assert handler_chain[0].value is _handler_one
    assert handler_chain[0].shadowed_by == "function"
    assert handler_chain[1].value is _handler_two
    assert handler_chain[1].shadowed_by is None

    debug_chain = merged.shadow_chains["debug"]
    assert [entry.scope for entry in debug_chain] == ["session", "module"]
    assert debug_chain[0].value is True
    assert debug_chain[0].shadowed_by == "module"
    assert debug_chain[1].value is False
    assert debug_chain[1].shadowed_by is None

    slow_chain = merged.shadow_chains["slow_callback_duration"]
    assert [(entry.scope, entry.value) for entry in slow_chain] == [
        ("session", 1.0),
        ("function", 2.5),
    ]
    assert slow_chain[0].shadowed_by == "function"


def test_merge_orders_layers_independently_of_input_order() -> None:
    merger = ExceptionHandlerConfigMerger(
        [
            _layer("function", "test_mod.py::test_x", debug=True),
            _layer("session", "session", debug=False),
        ]
    )

    merged = merger.merge()

    assert merged.debug is True
    debug_source = merged.sources["debug"]
    assert debug_source is not None
    assert debug_source.scope == "function"
    assert [entry.scope for entry in merged.shadow_chains["debug"]] == [
        "session",
        "function",
    ]


def test_merge_without_layers_leaves_fields_unset() -> None:
    merged = ExceptionHandlerConfigMerger([]).merge()

    assert merged.exception_handler is UNSET
    assert merged.debug is UNSET
    assert merged.slow_callback_duration is UNSET
    assert merged.sources == {
        "exception_handler": None,
        "debug": None,
        "slow_callback_duration": None,
    }
    assert merged.shadow_chains == {
        "exception_handler": (),
        "debug": (),
        "slow_callback_duration": (),
    }


def test_merge_rejects_unknown_scope() -> None:
    with pytest.raises(ExceptionConfigError, match="Unknown configuration scope"):
        ExceptionHandlerConfigMerger(
            [_layer("galaxy", "test_mod.py::test_x", debug=True)]
        )


def test_unknown_exception_handler_error_carries_location() -> None:
    merger = ExceptionHandlerConfigMerger(
        [_layer("function", "test_mod.py::test_x", exception_handler="nope")]
    )

    with pytest.raises(UnknownExceptionHandlerError) as exc_info:
        merger.merge()

    message = str(exc_info.value)
    assert "nope" in message
    assert "function" in message
    assert "test_mod.py::test_x" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("debug", "yes"),
        ("slow_callback_duration", -1.0),
        ("slow_callback_duration", True),
    ],
)
def test_invalid_setting_values_are_rejected_with_location(
    field: str, value: object
) -> None:
    merger = ExceptionHandlerConfigMerger(
        [_layer("module", "test_mod.py", **{field: value})]
    )

    with pytest.raises(ExceptionConfigError) as exc_info:
        merger.merge()

    assert "module" in str(exc_info.value)
    assert "test_mod.py" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Snapshot / apply / restore unit tests
# ---------------------------------------------------------------------------


def test_snapshot_is_immutable(fresh_loop: asyncio.AbstractEventLoop) -> None:
    fresh_loop.set_debug(True)
    fresh_loop.slow_callback_duration = 0.42
    fresh_loop.set_exception_handler(_handler_one)

    snapshot = snapshot_loop_config(fresh_loop, "test_mod.py::test_snapshot")

    assert isinstance(snapshot, LoopConfigSnapshot)
    assert snapshot.exception_handler is _handler_one
    assert snapshot.debug is True
    assert snapshot.slow_callback_duration == 0.42
    assert snapshot.loop_id == id(fresh_loop)
    assert snapshot.nodeid == "test_mod.py::test_snapshot"

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.debug = False  # type: ignore[misc]


def test_apply_and_restore_roundtrip(fresh_loop: asyncio.AbstractEventLoop) -> None:
    assert fresh_loop.get_exception_handler() is None
    assert fresh_loop.get_debug() is False
    assert fresh_loop.slow_callback_duration == 0.1

    merged = ExceptionHandlerConfigMerger(
        [
            _layer("module", "test_mod.py", debug=True, slow_callback_duration=0.02),
            _layer(
                "function",
                "test_mod.py::test_x",
                exception_handler=_handler_one,
            ),
        ]
    ).merge()
    snapshot = snapshot_loop_config(fresh_loop, "test_mod.py::test_x")
    applied = apply_loop_config(fresh_loop, merged, "test_mod.py::test_x")

    assert fresh_loop.get_debug() is True
    assert fresh_loop.slow_callback_duration == 0.02
    assert fresh_loop.get_exception_handler() is _handler_one

    restore_loop_config(fresh_loop, snapshot, applied)

    assert fresh_loop.get_exception_handler() is None
    assert fresh_loop.get_debug() is False
    assert fresh_loop.slow_callback_duration == 0.1


def test_apply_unknown_handler_is_atomic(fresh_loop: asyncio.AbstractEventLoop) -> None:
    merged = MergedExceptionHandlerConfig(
        exception_handler=object(),
        debug=UNSET,
        slow_callback_duration=UNSET,
        sources={},
        shadow_chains={},
    )

    with pytest.raises(UnknownExceptionHandlerError):
        apply_loop_config(fresh_loop, merged, "test_mod.py::test_x")

    # The loop must be completely untouched after a rejected application.
    assert fresh_loop.get_exception_handler() is None
    assert fresh_loop.get_debug() is False
    assert fresh_loop.slow_callback_duration == 0.1


def test_context_manager_restores_on_clean_exit(
    fresh_loop: asyncio.AbstractEventLoop,
) -> None:
    merged = ExceptionHandlerConfigMerger(
        [_layer("function", "test_mod.py::test_x", debug=True)]
    ).merge()

    with loop_exception_config(fresh_loop, merged, "test_mod.py::test_x") as snapshot:
        assert isinstance(snapshot, LoopConfigSnapshot)
        assert fresh_loop.get_debug() is True

    assert fresh_loop.get_debug() is False


def test_restore_conflict_preserves_main_exception(
    fresh_loop: asyncio.AbstractEventLoop,
) -> None:
    merged = ExceptionHandlerConfigMerger(
        [
            _layer(
                "function",
                "test_mod.py::test_x",
                exception_handler=_handler_one,
                debug=True,
            )
        ]
    ).merge()

    with pytest.raises(ExceptionConfigRestoreError) as exc_info:
        with loop_exception_config(fresh_loop, merged, "test_mod.py::test_x"):
            # Something outside the configuration layer reconfigures the loop.
            fresh_loop.set_exception_handler(_handler_two)
            raise RuntimeError("main test failure")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "main test failure"
    message = str(exc_info.value)
    assert "exception_handler" in message
    assert "test_mod.py::test_x" in message
    # The conflicting field is force-restored despite the reported conflict.
    assert fresh_loop.get_exception_handler() is None
    assert fresh_loop.get_debug() is False


def test_restore_collects_all_conflicts(
    fresh_loop: asyncio.AbstractEventLoop,
) -> None:
    merged = ExceptionHandlerConfigMerger(
        [
            _layer(
                "function",
                "test_mod.py::test_x",
                exception_handler=_handler_one,
                debug=True,
                slow_callback_duration=0.02,
            )
        ]
    ).merge()

    with pytest.raises(ExceptionConfigRestoreError) as exc_info:
        with loop_exception_config(fresh_loop, merged, "test_mod.py::test_x"):
            fresh_loop.set_exception_handler(_handler_two)
            fresh_loop.set_debug(False)
            fresh_loop.slow_callback_duration = 0.5

    message = str(exc_info.value)
    assert "exception_handler" in message
    assert "debug" in message
    assert "slow_callback_duration" in message
    # Every field was restored despite the conflicts.
    assert fresh_loop.get_exception_handler() is None
    assert fresh_loop.get_debug() is False
    assert fresh_loop.slow_callback_duration == 0.1


def test_restore_rejects_different_loop() -> None:
    loop_one = asyncio.new_event_loop()
    loop_two = asyncio.new_event_loop()
    try:
        merged = ExceptionHandlerConfigMerger(
            [_layer("function", "test_mod.py::test_x", debug=True)]
        ).merge()
        snapshot = snapshot_loop_config(loop_one, "test_mod.py::test_x")
        applied = apply_loop_config(loop_one, merged, "test_mod.py::test_x")

        with pytest.raises(ExceptionConfigRestoreError, match="different loop"):
            restore_loop_config(loop_two, snapshot, applied)
    finally:
        loop_one.close()
        loop_two.close()


def test_closed_loop_operations_raise(fresh_loop: asyncio.AbstractEventLoop) -> None:
    merged = ExceptionHandlerConfigMerger(
        [_layer("function", "test_mod.py::test_x", debug=True)]
    ).merge()
    snapshot = snapshot_loop_config(fresh_loop, "test_mod.py::test_x")
    applied = apply_loop_config(fresh_loop, merged, "test_mod.py::test_x")
    fresh_loop.close()

    with pytest.raises(ExceptionConfigError, match="closed event loop"):
        snapshot_loop_config(fresh_loop, "test_mod.py::test_x")
    with pytest.raises(ExceptionConfigError, match="closed"):
        apply_loop_config(fresh_loop, merged, "test_mod.py::test_x")
    with pytest.raises(ExceptionConfigRestoreError, match="closed"):
        restore_loop_config(fresh_loop, snapshot, applied)


# ---------------------------------------------------------------------------
# Pytest integration tests
# ---------------------------------------------------------------------------


def _run_pytest(pytester: Pytester, *args: str):
    # Explicitly disable a plugin that breaks inner pytest runs on some
    # machines. Blocking a non-installed plugin is a harmless no-op, so this
    # stays portable for CI environments.
    return pytester.runpytest("-p", "no:tornado", *args)


def test_nested_scope_settings_are_merged_and_restored(pytester: Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import asyncio

            import pytest

            pytest_plugins = "pytest_asyncio"


            def module_handler(loop, context):
                pass


            def function_handler(loop, context):
                pass


            pytestmark = pytest.mark.asyncio_exception_config(
                debug=True,
                slow_callback_duration=2.5,
                exception_handler=module_handler,
            )


            class TestClass:
                pytestmark = pytest.mark.asyncio_exception_config(debug=False)

                @pytest.mark.asyncio_exception_config(
                    exception_handler=function_handler
                )
                @pytest.mark.asyncio
                async def test_function_layer_wins(self):
                    loop = asyncio.get_running_loop()
                    # The class layer shadows the module debug setting.
                    assert loop.get_debug() is False
                    # Unshadowed module settings remain effective.
                    assert loop.slow_callback_duration == 2.5
                    # The function layer shadows the module handler.
                    assert loop.get_exception_handler() is function_handler

                @pytest.mark.asyncio
                async def test_class_and_module_layers_only(self):
                    loop = asyncio.get_running_loop()
                    assert loop.get_debug() is False
                    assert loop.slow_callback_duration == 2.5
                    assert loop.get_exception_handler() is module_handler
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(passed=2)


def test_shared_loop_is_restored_between_tests(pytester: Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import asyncio

            import pytest

            pytest_plugins = "pytest_asyncio"

            shared = {}


            def custom_handler(loop, context):
                shared["handler_called"] = True


            @pytest.mark.asyncio_exception_config(
                exception_handler=custom_handler,
                debug=True,
                slow_callback_duration=0.01,
            )
            @pytest.mark.asyncio(loop_scope="session")
            async def test_configures_shared_loop():
                loop = asyncio.get_running_loop()
                shared["loop"] = loop
                assert loop.get_debug() is True
                assert loop.get_exception_handler() is custom_handler
                assert loop.slow_callback_duration == 0.01


            @pytest.mark.asyncio(loop_scope="session")
            async def test_shared_loop_is_back_to_its_snapshot():
                loop = asyncio.get_running_loop()
                assert loop is shared["loop"]
                assert loop.get_debug() is False
                assert loop.get_exception_handler() is None
                assert loop.slow_callback_duration == 0.1
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(passed=2)


def test_setup_failure_with_restore_conflict_keeps_main_exception(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import asyncio

            import pytest
            import pytest_asyncio

            pytest_plugins = "pytest_asyncio"


            def configured_handler(loop, context):
                pass


            def external_handler(loop, context):
                pass


            @pytest_asyncio.fixture(loop_scope="module")
            async def meddling_fixture():
                loop = asyncio.get_running_loop()
                loop.set_exception_handler(external_handler)
                raise RuntimeError("boom during setup")


            @pytest.mark.asyncio_exception_config(
                exception_handler=configured_handler, debug=True
            )
            @pytest.mark.asyncio(loop_scope="module")
            async def test_setup_fails(meddling_fixture):
                pass


            @pytest.mark.asyncio(loop_scope="module")
            async def test_loop_was_restored_after_setup_failure():
                loop = asyncio.get_running_loop()
                assert loop.get_exception_handler() is None
                assert loop.get_debug() is False
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(passed=1, errors=1)
    stdout = result.stdout.str()
    assert "boom during setup" in stdout
    assert "restore conflict" in stdout
    assert "exception_handler" in stdout


def test_setup_failure_without_conflict_restores_cleanly(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import asyncio

            import pytest
            import pytest_asyncio

            pytest_plugins = "pytest_asyncio"


            @pytest_asyncio.fixture(loop_scope="module")
            async def failing_fixture():
                raise RuntimeError("clean boom in setup")


            @pytest.mark.asyncio_exception_config(debug=True)
            @pytest.mark.asyncio(loop_scope="module")
            async def test_setup_fails(failing_fixture):
                pass


            @pytest.mark.asyncio(loop_scope="module")
            async def test_loop_was_restored():
                loop = asyncio.get_running_loop()
                assert loop.get_debug() is False
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(passed=1, errors=1)
    stdout = result.stdout.str()
    assert "clean boom in setup" in stdout
    assert "restore conflict" not in stdout


def test_unknown_handler_marker_reports_location(pytester: Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import pytest

            pytest_plugins = "pytest_asyncio"


            @pytest.mark.asyncio_exception_config(exception_handler=123)
            @pytest.mark.asyncio
            async def test_unknown_handler():
                pass
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(errors=1)
    stdout = result.stdout.str()
    assert "Unknown exception handler" in stdout
    assert "123" in stdout
    assert "function" in stdout
    assert "test_unknown_handler" in stdout


@pytest.mark.parametrize(
    "marker_call",
    [
        "pytest.mark.asyncio_exception_config(foo=1)",
        "pytest.mark.asyncio_exception_config()",
        "pytest.mark.asyncio_exception_config(True)",
    ],
)
def test_invalid_marker_usage_is_rejected(pytester: Pytester, marker_call: str) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent(f"""\
            import pytest

            pytest_plugins = "pytest_asyncio"


            @{marker_call}
            @pytest.mark.asyncio
            async def test_invalid_marker():
                pass
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(errors=1)
    assert "asyncio_exception_config" in result.stdout.str()


def test_tests_without_marker_are_completely_untouched(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import asyncio

            import pytest

            pytest_plugins = "pytest_asyncio"


            @pytest.mark.asyncio
            async def test_default_loop_config():
                loop = asyncio.get_running_loop()
                assert loop.get_exception_handler() is None
                assert loop.get_debug() is False
                assert loop.slow_callback_duration == 0.1
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(passed=1)


def test_skip_during_setup_still_restores_the_loop(pytester: Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import asyncio

            import pytest
            import pytest_asyncio

            pytest_plugins = "pytest_asyncio"


            @pytest_asyncio.fixture(loop_scope="module")
            async def skipping_fixture():
                pytest.skip("skipping from a fixture")


            @pytest.mark.asyncio_exception_config(debug=True)
            @pytest.mark.asyncio(loop_scope="module")
            async def test_skips(skipping_fixture):
                pass


            @pytest.mark.asyncio(loop_scope="module")
            async def test_loop_restored_after_skip():
                loop = asyncio.get_running_loop()
                assert loop.get_debug() is False
            """)
    )
    result = _run_pytest(pytester)
    result.assert_outcomes(skipped=1, passed=1)


def test_marker_overrides_cli_debug_and_shared_loop_is_restored(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent("""\
            import asyncio

            import pytest

            pytest_plugins = "pytest_asyncio"

            shared = {}


            @pytest.mark.asyncio_exception_config(debug=False)
            @pytest.mark.asyncio(loop_scope="module")
            async def test_marker_disables_debug():
                loop = asyncio.get_running_loop()
                shared["loop"] = loop
                assert loop.get_debug() is False


            @pytest.mark.asyncio(loop_scope="module")
            async def test_shared_loop_returns_to_cli_debug_baseline():
                loop = asyncio.get_running_loop()
                assert loop is shared["loop"]
                assert loop.get_debug() is True
            """)
    )
    result = _run_pytest(pytester, "--asyncio-debug")
    result.assert_outcomes(passed=2)

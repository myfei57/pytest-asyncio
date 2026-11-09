"""Tests for the read-only coroutine-wrapper provenance chain."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
from textwrap import dedent
from typing import Any, cast

import pytest
import pytest_asyncio
import pytest_asyncio.plugin as pao_plugin
from pytest import Pytester


def test_empty_chain_for_objects_never_wrapped() -> None:
    async def standalone_coro() -> None:
        pass

    def standalone_func() -> None:
        pass

    assert pytest_asyncio.get_coroutine_wrapper_chain(object()) == ()
    assert pytest_asyncio.get_coroutine_wrapper_chain(standalone_coro) == ()
    assert pytest_asyncio.get_coroutine_wrapper_chain(standalone_func) == ()


def test_repeated_wrapping_is_rejected_and_original_retained() -> None:
    invocations = 0

    async def coro() -> None:
        nonlocal invocations
        invocations += 1

    record = pytest_asyncio.CoroutineWrapperSource(
        nodeid="test_module.py::test_coro",
        wrapper_version=pytest_asyncio.COROUTINE_WRAPPER_VERSION,
        loop_scope="function",
    )

    class _FakeItem:
        nodeid = record.nodeid
        _wrapper_source_chain = (record,)

    fake_item = cast(pao_plugin.PytestAsyncioFunction, _FakeItem())
    with asyncio.Runner() as runner:
        wrapped = pao_plugin._synchronize_coroutine(
            coro,
            runner,
            contextvars.copy_context(),
            item=fake_item,
        )

        chain = pytest_asyncio.get_coroutine_wrapper_chain(wrapped)
        assert len(chain) == 1
        assert chain == (record,)
        # The raw coroutine itself is never stamped.
        assert pytest_asyncio.get_coroutine_wrapper_chain(coro) == ()

        with pytest.raises(pytest_asyncio.CoroutineWrapperError):
            pao_plugin._synchronize_coroutine(
                cast(Any, wrapped),
                runner,
                contextvars.copy_context(),
                item=fake_item,
            )

    # Rejection retains the original object: no second link is ever added
    # and the wrapped coroutine was never executed.
    assert pytest_asyncio.get_coroutine_wrapper_chain(wrapped) == chain
    assert not hasattr(coro, pao_plugin._WRAPPER_SOURCE_CHAIN_ATTR)
    assert invocations == 0
    assert issubclass(
        pytest_asyncio.CoroutineWrapperError, pao_plugin.PytestAsyncioError
    )


def test_chain_and_records_are_read_only() -> None:
    record = pytest_asyncio.CoroutineWrapperSource(
        nodeid="test_module.py::test_coro",
        wrapper_version=pytest_asyncio.COROUTINE_WRAPPER_VERSION,
        loop_scope="function",
    )
    chain = (record,)

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.loop_scope = "session"  # type: ignore[misc]
    with pytest.raises(TypeError):
        chain[0] = record  # type: ignore[index]
    assert not hasattr(chain, "append")
    assert not hasattr(chain, "extend")
    assert not hasattr(chain, "clear")


def test_parametrized_nodes_recorded_without_running_tests(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent(
            """
        import pytest_asyncio

        def pytest_collection_modifyitems(items):
            assert len(items) == 2
            for item in items:
                chain = pytest_asyncio.get_coroutine_wrapper_chain(item)
                assert isinstance(chain, tuple)
                assert len(chain) == 1
                source = chain[0]
                assert source.nodeid == item.nodeid
                assert "test_param[" in source.nodeid
                assert (
                    source.wrapper_version
                    == pytest_asyncio.COROUTINE_WRAPPER_VERSION
                )
                assert source.loop_scope == "function"
        """
        )
    )
    pytester.makepyfile(
        dedent(
            """
        import pytest

        @pytest.mark.asyncio
        @pytest.mark.parametrize("x", [1, 2])
        async def test_param(x):
            open("ran.txt", "w").write(str(x))
        """
        )
    )
    result = pytester.runpytest("--collect-only", "--asyncio-mode=strict")
    assert result.ret == 0
    # Reading the chain must not execute any test.
    assert not pytester.path.joinpath("ran.txt").exists()


def test_loop_scope_recorded_from_marker_and_default(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent(
            """
        import pytest_asyncio

        def pytest_collection_modifyitems(items):
            scopes = {
                item.name: pytest_asyncio.get_coroutine_wrapper_chain(item)[
                    0
                ].loop_scope
                for item in items
            }
            assert scopes == {
                "test_module_scoped": "module",
                "test_function_scoped": "function",
            }
        """
        )
    )
    pytester.makepyfile(
        dedent(
            """
        import pytest

        @pytest.mark.asyncio(loop_scope="module")
        async def test_module_scoped():
            pass

        @pytest.mark.asyncio
        async def test_function_scoped():
            pass
        """
        )
    )
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_stacked_decorators_keep_a_single_chain_link(pytester: Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent(
            """
        import pytest
        import pytest_asyncio

        def pytest_collection_modifyitems(items):
            for item in items:
                chain = pytest_asyncio.get_coroutine_wrapper_chain(item)
                assert len(chain) == 1
                assert chain[0].nodeid == item.nodeid

        @pytest.hookimpl(hookwrapper=True)
        def pytest_pyfunc_call(pyfuncitem):
            item_chain = pytest_asyncio.get_coroutine_wrapper_chain(pyfuncitem)
            wrapper_chain = pytest_asyncio.get_coroutine_wrapper_chain(
                pyfuncitem.obj
            )
            assert len(wrapper_chain) == 1
            assert wrapper_chain == item_chain
            # The raw underlying coroutine was never stamped.
            assert pytest_asyncio.get_coroutine_wrapper_chain(
                pyfuncitem.obj.__wrapped__
            ) == ()
            yield
        """
        )
    )
    pytester.makepyfile(
        dedent(
            """
        import functools

        import pytest

        def preserve_async(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)

            return wrapper

        @preserve_async
        @pytest.mark.asyncio
        @pytest.mark.parametrize("x", [1, 2])
        async def test_decorated(x):
            assert x in (1, 2)
        """
        )
    )
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_auto_mode_records_chain_with_default_loop_scope(
    pytester: Pytester,
) -> None:
    pytester.makeini(
        dedent(
            """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        asyncio_default_test_loop_scope = module
        asyncio_mode = auto
        """
        )
    )
    pytester.makeconftest(
        dedent(
            """
        import pytest_asyncio

        def pytest_collection_modifyitems(items):
            assert len(items) == 2
            for item in items:
                assert item.get_closest_marker("asyncio") is not None
                chain = pytest_asyncio.get_coroutine_wrapper_chain(item)
                assert len(chain) == 1
                source = chain[0]
                assert source.nodeid == item.nodeid
                assert (
                    source.wrapper_version
                    == pytest_asyncio.COROUTINE_WRAPPER_VERSION
                )
                assert source.loop_scope == "module"
        """
        )
    )
    pytester.makepyfile(
        dedent(
            """
        async def test_auto():
            pass


        class TestClass:
            @staticmethod
            async def test_static():
                pass
        """
        )
    )
    result = pytester.runpytest("--asyncio-mode=auto")
    result.assert_outcomes(passed=2)


def test_collection_marker_error_does_not_leak_chain(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent(
            """
        import pytest_asyncio

        def pytest_collection_modifyitems(items):
            # Raw collected functions never carry wrapper provenance.
            leaked = [
                item.nodeid
                for item in items
                if pytest_asyncio.get_coroutine_wrapper_chain(item.obj)
            ]
            assert not leaked
        """
        )
    )
    pytester.makepyfile(
        dedent(
            """
        import pytest

        @pytest.mark.asyncio(loop_scope="invalid_scope")
        async def test_bad_marker():
            pass
        """
        )
    )
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["test_*::test_bad_marker: mark.asyncio got an unexpected loop scope*"]
    )
    result.stdout.no_fnmatch_line("*CoroutineWrapperError*")


def test_collection_import_error_does_not_leak_chain(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makepyfile(
        dedent(
            """
        raise RuntimeError("boom during import")
        """
        )
    )
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*RuntimeError: boom during import*"])
    result.stdout.no_fnmatch_line("*CoroutineWrapperError*")

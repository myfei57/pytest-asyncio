from __future__ import annotations

import asyncio
import dataclasses
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent

import pytest
from pytest import Pytester

from pytest_asyncio.loop_factory_negotiation import (
    NEGOTIABLE_SCOPES,
    LoopFactoryCapability,
    LoopFactoryConflictError,
    LoopFactoryDowngradeError,
    LoopFactoryLoadError,
    LoopFactoryNegotiator,
    LoopFactorySnapshot,
    as_loop_factory_capability,
)


def _factory_a() -> asyncio.AbstractEventLoop:
    return asyncio.new_event_loop()


def _factory_b() -> asyncio.AbstractEventLoop:
    return asyncio.new_event_loop()


def _capability(
    name: str = "custom",
    *,
    factory=_factory_a,
    scopes: frozenset[str] = NEGOTIABLE_SCOPES,
    supports_debug: bool = True,
    version: int = 1,
) -> LoopFactoryCapability:
    return LoopFactoryCapability(
        name=name,
        factory=factory,
        scopes=scopes,
        supports_debug=supports_debug,
        version=version,
    )


# ---------------------------------------------------------------------------
# Immutable capabilities and snapshots
# ---------------------------------------------------------------------------


def test_capability_normalizes_empty_scope_set_to_all_scopes() -> None:
    capability = _capability(scopes=frozenset())
    assert capability.scopes == NEGOTIABLE_SCOPES


def test_capability_is_immutable() -> None:
    capability = _capability()
    with pytest.raises(dataclasses.FrozenInstanceError):
        capability.version = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "non-empty string name"),
        ({"name": "x", "factory": 1}, "must be a callable"),
        ({"name": "x", "scopes": frozenset({"bogus"})}, "unsupported loop scope"),
        ({"name": "x", "version": 0}, "version must be >= 1"),
        ({"name": "x", "version": True}, "version must be an integer"),
    ],
)
def test_capability_validates_arguments(kwargs: dict, message: str) -> None:
    with pytest.raises(LoopFactoryLoadError, match=message):
        _capability(**kwargs)


def test_capability_compatibility_checks_scope_and_debug() -> None:
    capability = _capability(scopes=frozenset({"module"}), supports_debug=False)
    assert capability.compatible_with("module")
    assert not capability.compatible_with("function")
    assert not capability.compatible_with("module", debug=True)


def test_snapshot_is_immutable() -> None:
    snapshot = LoopFactorySnapshot("key", [_capability()])
    with pytest.raises(TypeError):
        snapshot.factories["other"] = _capability("other")  # type: ignore[index]
    with pytest.raises(AttributeError):
        snapshot.key = "other"  # type: ignore[misc]


def test_snapshot_rejects_duplicate_and_invalid_entries() -> None:
    with pytest.raises(LoopFactoryLoadError, match="Duplicate loop factory name"):
        LoopFactorySnapshot("key", [_capability("same"), _capability("same")])
    with pytest.raises(LoopFactoryLoadError):
        LoopFactorySnapshot("key", [_factory_a])  # type: ignore[list-item]


def test_snapshot_only_exposes_compatible_factories_for_scope() -> None:
    module_only = _capability(
        "module_only",
        factory=_factory_b,
        scopes=frozenset({"module"}),
        supports_debug=False,
    )
    universal = _capability("universal")
    snapshot = LoopFactorySnapshot(("module", "test_mod.py"), [module_only, universal])

    assert {cap.name for cap in snapshot.compatible("function")} == {"universal"}
    assert {cap.name for cap in snapshot.compatible("module")} == {
        "module_only",
        "universal",
    }
    assert {cap.name for cap in snapshot.compatible("module", debug=True)} == {
        "universal"
    }


def test_snapshot_equality_is_independent_of_insertion_order() -> None:
    first = LoopFactorySnapshot("key", [_capability("a"), _capability("b")])
    second = LoopFactorySnapshot("key", [_capability("b"), _capability("a")])
    assert first == second
    assert hash(first) == hash(second)


def test_bare_callable_is_coerced_to_default_capability() -> None:
    capability = as_loop_factory_capability("plain", _factory_a)
    assert capability.name == "plain"
    assert capability.factory is _factory_a
    assert capability.scopes == NEGOTIABLE_SCOPES
    assert capability.supports_debug is True
    assert capability.version == 1


def test_capability_registered_under_mismatched_name_is_rejected() -> None:
    capability = _capability("real")
    with pytest.raises(LoopFactoryLoadError, match="registered under"):
        as_loop_factory_capability("alias", capability)


# ---------------------------------------------------------------------------
# Negotiation: same-name factories, versions, failure isolation
# ---------------------------------------------------------------------------


def test_negotiate_publishes_immutable_snapshot() -> None:
    negotiator = LoopFactoryNegotiator()
    snapshot = negotiator.negotiate(
        ("function", "test_x.py::test_a"), [{"custom": _factory_a}]
    )
    assert snapshot.names == ("custom",)
    assert negotiator.effective_snapshot(snapshot.key) is snapshot


def test_negotiation_is_idempotent_for_parametrized_nodes() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("module", "test_x.py")
    first = negotiator.negotiate(key, [{"custom": _factory_a}])
    # Every parametrized variant of one node re-enters the same scope and
    # reloads the same factories; that must not raise or change anything.
    second = negotiator.negotiate(key, [{"custom": _factory_a}])
    assert first == second
    assert negotiator.effective_snapshot(key) == first


def test_same_name_factory_with_higher_version_overrides() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("module", "test_x.py")
    first = negotiator.negotiate(key, [{"custom": _capability(version=1)}])
    upgraded = _capability(factory=_factory_b, version=2)
    second = negotiator.negotiate(key, [{"custom": upgraded}])

    assert second["custom"].version == 2
    assert second["custom"].factory is _factory_b
    # Publishing a new snapshot does not mutate the already effective object.
    assert first["custom"].version == 1
    assert first["custom"].factory is _factory_a


def test_same_name_version_downgrade_is_rejected() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("module", "test_x.py")
    negotiator.negotiate(key, [{"custom": _capability(version=2)}])
    with pytest.raises(LoopFactoryDowngradeError, match="older version"):
        negotiator.negotiate(key, [{"custom": _capability(version=1)}])
    effective = negotiator.effective_snapshot(key)
    assert effective is not None
    assert effective["custom"].version == 2


def test_same_name_same_version_with_different_capabilities_conflicts() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("module", "test_x.py")
    negotiator.negotiate(key, [{"custom": _capability(version=1)}])
    conflicting = _capability(factory=_factory_b, version=1)
    with pytest.raises(LoopFactoryConflictError):
        negotiator.negotiate(key, [{"custom": conflicting}])
    effective = negotiator.effective_snapshot(key)
    assert effective is not None
    assert effective["custom"].factory is _factory_a


def test_duplicate_name_within_one_load_is_rejected() -> None:
    negotiator = LoopFactoryNegotiator()
    with pytest.raises(LoopFactoryLoadError, match="Duplicate loop factory name"):
        negotiator.negotiate(
            "key",
            [
                {"custom": _factory_a},
                {"custom": _factory_b},
            ],
        )
    assert negotiator.effective_snapshot("key") is None


def test_failing_provider_does_not_pollute_effective_snapshot() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("module", "test_x.py")
    negotiator.negotiate(key, [{"good": _capability("good")}])

    def failing_provider() -> Mapping[str, object]:
        raise RuntimeError("factory plugin exploded")

    with pytest.raises(LoopFactoryLoadError):
        negotiator.negotiate(
            key,
            [
                {"new": _capability("new")},
                failing_provider,
            ],
        )
    effective = negotiator.effective_snapshot(key)
    assert effective is not None
    assert effective.names == ("good",)


def test_invalid_capability_in_batch_publishes_nothing() -> None:
    negotiator = LoopFactoryNegotiator()
    with pytest.raises(LoopFactoryLoadError):
        negotiator.negotiate(
            "fresh",
            [
                {"good": _factory_a},
                {"bad": 1},  # type: ignore[dict-item]
            ],
        )
    assert negotiator.effective_snapshot("fresh") is None


def test_non_mapping_provider_result_is_a_load_failure() -> None:
    negotiator = LoopFactoryNegotiator()
    with pytest.raises(LoopFactoryLoadError, match="must return a mapping"):
        negotiator.negotiate("key", [lambda: ["not", "a", "mapping"]])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Concurrent collection
# ---------------------------------------------------------------------------


def test_concurrent_negotiation_of_same_factories_is_consistent() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("session", "")
    barrier = threading.Barrier(8)

    def load() -> LoopFactorySnapshot:
        barrier.wait()
        return negotiator.negotiate(key, [{"custom": _factory_a}])

    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(lambda _: load(), range(8)))

    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    effective = negotiator.effective_snapshot(key)
    assert effective is not None
    assert effective.names == ("custom",)
    assert effective["custom"].version == 1


def test_concurrent_negotiation_merges_distinct_factories() -> None:
    negotiator = LoopFactoryNegotiator()
    barrier = threading.Barrier(8)

    def load(index: int) -> LoopFactorySnapshot:
        barrier.wait()
        return negotiator.negotiate(
            ("session", ""),
            [{f"factory_{index}": _factory_a}],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(load, range(8)))

    effective = negotiator.effective_snapshot(("session", ""))
    assert effective is not None
    assert len(effective) == 8
    assert all(len(snapshot) <= 8 for snapshot in snapshots)


def test_concurrent_downgrades_never_replace_effective_snapshot() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("session", "")
    negotiator.negotiate(key, [{"custom": _capability(version=2)}])
    barrier = threading.Barrier(8)

    def downgrade() -> None:
        barrier.wait()
        with pytest.raises(LoopFactoryDowngradeError):
            negotiator.negotiate(key, [{"custom": _capability(version=1)}])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: downgrade(), range(8)))

    effective = negotiator.effective_snapshot(key)
    assert effective is not None
    assert effective["custom"].version == 2


def test_concurrent_failures_leave_effective_snapshot_untouched() -> None:
    negotiator = LoopFactoryNegotiator()
    key = ("session", "")
    negotiator.negotiate(key, [{"good": _capability("good", version=2)}])
    start = threading.Barrier(6)

    def worker(index: int) -> None:
        start.wait()
        try:
            if index % 2:

                def fail() -> Mapping[str, object]:
                    raise RuntimeError("boom")

                negotiator.negotiate(key, [fail])
            else:
                negotiator.negotiate(key, [{"good": _capability("good", version=2)}])
        except LoopFactoryLoadError:
            pass

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, range(6)))

    effective = negotiator.effective_snapshot(key)
    assert effective is not None
    assert effective.names == ("good",)
    assert effective["good"].version == 2


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------


def test_capability_hook_parametrizes_and_runs_custom_loops(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent("""\
        import asyncio

        import pytest_asyncio

        class LoopA(asyncio.SelectorEventLoop):
            pass

        class LoopB(asyncio.SelectorEventLoop):
            pass

        def pytest_asyncio_loop_factories(config, item):
            return {
                "loop_a": pytest_asyncio.LoopFactoryCapability(
                    name="loop_a", factory=LoopA
                ),
                "loop_b": pytest_asyncio.LoopFactoryCapability(
                    name="loop_b", factory=LoopB
                ),
            }
        """)
    )
    pytester.makepyfile(
        dedent("""\
        import asyncio
        import pytest

        pytest_plugins = "pytest_asyncio"

        @pytest.mark.parametrize("value", [1, 2])
        @pytest.mark.asyncio
        async def test_runs_on_negotiated_factory(value):
            loop_name = type(asyncio.get_running_loop()).__name__
            assert loop_name in ("LoopA", "LoopB")
        """)
    )
    result = pytester.runpytest("--asyncio-mode=strict")
    # Two factories times two user parameters: four parametrized nodes.
    result.assert_outcomes(passed=4)


def test_node_entering_scope_rejects_incompatible_factory(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent("""\
        import asyncio

        import pytest_asyncio

        class LimitedLoop(asyncio.SelectorEventLoop):
            pass

        class UniversalLoop(asyncio.SelectorEventLoop):
            pass

        def pytest_asyncio_loop_factories(config, item):
            return {
                "limited": pytest_asyncio.LoopFactoryCapability(
                    name="limited",
                    factory=LimitedLoop,
                    scopes=frozenset({"function"}),
                ),
                "universal": pytest_asyncio.LoopFactoryCapability(
                    name="universal", factory=UniversalLoop
                ),
            }
        """)
    )
    pytester.makepyfile(
        dedent("""\
        import asyncio
        import pytest

        pytest_plugins = "pytest_asyncio"

        @pytest.mark.asyncio(
            loop_scope="module", loop_factories=["limited", "universal"]
        )
        async def test_limited_is_skipped():
            assert (
                type(asyncio.get_running_loop()).__name__ == "UniversalLoop"
            )
        """)
    )
    result = pytester.runpytest("--asyncio-mode=strict", "-rs")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*SKIPPED*does not support loop scope 'module'*"])


def test_debug_incompatible_factory_is_rejected(pytester: Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent("""\
        import asyncio

        import pytest_asyncio

        class DebugLoop(asyncio.SelectorEventLoop):
            pass

        def pytest_asyncio_loop_factories(config, item):
            return {
                "nodebug": pytest_asyncio.LoopFactoryCapability(
                    name="nodebug",
                    factory=DebugLoop,
                    supports_debug=False,
                ),
            }
        """)
    )
    pytester.makepyfile(
        dedent("""\
        import pytest

        pytest_plugins = "pytest_asyncio"

        @pytest.mark.asyncio
        async def test_runs_without_debug():
            assert True
        """)
    )
    skipped = pytester.runpytest("--asyncio-mode=strict", "--asyncio-debug", "-rs")
    skipped.assert_outcomes(skipped=1)
    skipped.stdout.fnmatch_lines(["*SKIPPED*does not support asyncio debug mode*"])

    passed = pytester.runpytest("--asyncio-mode=strict")
    passed.assert_outcomes(passed=1)


def test_same_name_factory_version_upgrade_takes_effect(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent("""\
        import asyncio

        import pytest_asyncio

        class LoopV1(asyncio.SelectorEventLoop):
            pass

        class LoopV2(asyncio.SelectorEventLoop):
            pass

        def pytest_asyncio_loop_factories(config, item):
            if item.name.endswith("v1"):
                loop_cls, version = LoopV1, 1
            else:
                loop_cls, version = LoopV2, 2
            return {
                "custom": pytest_asyncio.LoopFactoryCapability(
                    name="custom", factory=loop_cls, version=version
                ),
            }
        """)
    )
    pytester.makepyfile(
        dedent("""\
        import asyncio
        import pytest

        pytest_plugins = "pytest_asyncio"

        @pytest.mark.asyncio(loop_scope="module")
        async def test_v1():
            assert type(asyncio.get_running_loop()).__name__ == "LoopV1"

        @pytest.mark.asyncio(loop_scope="module")
        async def test_v2():
            assert type(asyncio.get_running_loop()).__name__ == "LoopV2"
        """)
    )
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=2)


def test_same_name_factory_version_downgrade_errors_without_pollution(
    pytester: Pytester,
) -> None:
    pytester.makeini(
        "[pytest]\nasyncio_default_fixture_loop_scope = function\n"
        "asyncio_default_test_loop_scope = session"
    )
    pytester.makeconftest(
        dedent("""\
        import asyncio

        import pytest_asyncio

        class LoopV1(asyncio.SelectorEventLoop):
            pass

        class LoopV2(asyncio.SelectorEventLoop):
            pass

        def pytest_asyncio_loop_factories(config, item):
            if item.name.endswith("v2"):
                loop_cls, version = LoopV2, 2
            else:
                loop_cls, version = LoopV1, 1
            return {
                "custom": pytest_asyncio.LoopFactoryCapability(
                    name="custom", factory=loop_cls, version=version
                ),
            }
        """)
    )
    pytester.makepyfile(
        test_new=dedent("""\
            import pytest

            pytest_plugins = "pytest_asyncio"

            @pytest.mark.asyncio
            async def test_v2():
                assert True
            """),
        test_old=dedent("""\
            import pytest

            pytest_plugins = "pytest_asyncio"

            @pytest.mark.asyncio
            async def test_v1():
                assert True
            """),
    )
    # Both files share the session scope boundary: the older version must be
    # rejected while the already effective v2 snapshot stays usable.
    result = pytester.runpytest(
        "test_new.py",
        "test_old.py",
        "--asyncio-mode=strict",
        "--continue-on-collection-errors",
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*Refusing to replace loop factory*older version*"])


def test_failed_factory_load_does_not_pollute_effective_snapshot(
    pytester: Pytester,
) -> None:
    pytester.makeini(
        "[pytest]\nasyncio_default_fixture_loop_scope = function\n"
        "asyncio_default_test_loop_scope = session"
    )
    pytester.makeconftest(
        dedent("""\
        import asyncio

        import pytest
        import pytest_asyncio
        from pytest_asyncio.plugin import _loop_factory_negotiator_key

        class GoodLoop(asyncio.SelectorEventLoop):
            pass

        def pytest_asyncio_loop_factories(config, item):
            if item.name.endswith("good"):
                return {
                    "good": pytest_asyncio.LoopFactoryCapability(
                        name="good", factory=GoodLoop
                    )
                }
            raise RuntimeError("factory provider failed to load")

        @pytest.hookimpl(hookwrapper=True)
        def pytest_generate_tests(metafunc):
            try:
                outcome = yield
                outcome.get_result()
            except BaseException:
                negotiator = metafunc.config.stash[
                    _loop_factory_negotiator_key
                ]
                names = [
                    snapshot.names
                    for key in negotiator.effective_keys()
                    if (snapshot := negotiator.effective_snapshot(key))
                    is not None
                ]
                with open("effective_snapshots", "w", encoding="utf-8") as handle:
                    handle.write(repr(names))
                raise
        """)
    )
    pytester.makepyfile(
        test_good=dedent("""\
            import asyncio
            import pytest

            pytest_plugins = "pytest_asyncio"

            @pytest.mark.asyncio
            async def test_good():
                assert type(asyncio.get_running_loop()).__name__ == "GoodLoop"
            """),
        test_bad=dedent("""\
            import pytest

            pytest_plugins = "pytest_asyncio"

            @pytest.mark.asyncio
            async def test_bad():
                assert True
            """),
    )
    result = pytester.runpytest(
        "test_good.py",
        "test_bad.py",
        "--asyncio-mode=strict",
        "--continue-on-collection-errors",
    )
    result.assert_outcomes(passed=1, errors=1)
    recorded = (pytester.path / "effective_snapshots").read_text(encoding="utf-8")
    assert "good" in recorded
    assert "bad" not in recorded


def test_concurrent_collection_negotiates_consistent_snapshot(
    pytester: Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function")
    pytester.makeconftest(
        dedent("""\
        import asyncio
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import pytest
        from pytest_asyncio.plugin import (
            _loop_factory_negotiator_key,
            _scope_owner_nodeid,
        )

        class CustomLoop(asyncio.SelectorEventLoop):
            pass

        def pytest_asyncio_loop_factories(config, item):
            return {"custom": CustomLoop}

        @pytest.hookimpl(tryfirst=True)
        def pytest_generate_tests(metafunc):
            if not metafunc.definition.get_closest_marker("asyncio"):
                return
            negotiator = metafunc.config.stash[_loop_factory_negotiator_key]
            scope = "function"
            key = (
                scope,
                _scope_owner_nodeid(metafunc.definition, scope),
            )
            barrier = threading.Barrier(8)

            def negotiate():
                barrier.wait()
                return negotiator.negotiate(key, [{"custom": CustomLoop}])

            with ThreadPoolExecutor(max_workers=8) as pool:
                snapshots = list(pool.map(lambda _: negotiate(), range(8)))
            assert all(snapshot == snapshots[0] for snapshot in snapshots)
            assert snapshots[0].names == ("custom",)
        """)
    )
    pytester.makepyfile(
        dedent("""\
        import pytest

        pytest_plugins = "pytest_asyncio"

        @pytest.mark.asyncio
        async def test_collected_concurrently():
            assert True
        """)
    )
    result = pytester.runpytest("--asyncio-mode=strict")
    result.assert_outcomes(passed=1)

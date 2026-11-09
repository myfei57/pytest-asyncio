=====
Hooks
=====

``pytest_asyncio_loop_factories``
=================================

This hook returns a mapping from factory name strings to event loop factory callables for the current test item.

By default, each pytest-asyncio test is run once per configured factory. Tests managed by other async plugins are unaffected. Synchronous tests are not parametrized. The configured loop scope still determines how long each event loop instance is kept alive.

Factories should be callables without required parameters and should return an ``asyncio.AbstractEventLoop`` instance. The effective hook result must be a non-empty mapping of non-empty string names to callables.

When multiple ``pytest_asyncio_loop_factories`` implementations are present, pytest-asyncio uses the first non-``None`` result in pytest's hook dispatch order.

When the hook is defined, async tests are parametrized via ``pytest.metafunc.parametrize``, and mapping keys are used as test IDs. For example, a test ``test_example`` with an event loop factory key ``foo`` will appear as ``test_example[foo]`` in test output.

Capability negotiation snapshots
=================================

A mapping value may be either a plain callable or a :class:`pytest_asyncio.LoopFactoryCapability`. Plain callables are treated as factories that support every loop scope, support asyncio debug mode, and use capability version 1.

A ``LoopFactoryCapability`` advertises the factory name, its callable, the loop scopes it supports (an empty set means all scopes), whether it supports asyncio debug mode, and a monotonically increasing version. When a test node enters a loop scope, only compatible factories are accepted: the factory must support that scope and, when ``--asyncio-debug`` is active, it must support debug mode. Incompatible or unavailable factories are reported as skipped with an explanatory reason.

The capabilities negotiated for each loop scope boundary are published as immutable :class:`pytest_asyncio.LoopFactorySnapshot` objects by the session-wide :class:`pytest_asyncio.LoopFactoryNegotiator`. The rules are:

* a same-name factory with a higher version overrides the effective factory, while a version downgrade or a same-version capability conflict is rejected and leaves the effective snapshot untouched;
* failing factory providers or invalid capabilities discard the whole negotiation before it is published, so snapshots that are already effective are never polluted;
* negotiation is serialized, so concurrent collection threads can negotiate the same scope boundary safely.

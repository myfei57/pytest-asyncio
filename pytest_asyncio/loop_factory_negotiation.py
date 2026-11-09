"""
Capability negotiation for asyncio event loop factories.

This module turns the named loop factories returned by the
``pytest_asyncio_loop_factories`` hook into immutable *negotiation snapshots*.

A factory advertises its capabilities through :class:`LoopFactoryCapability`:
its name, the loop scopes it supports, whether it supports asyncio debug mode,
and a monotonically increasing version. When a test node enters a loop scope,
the node only accepts factories whose capabilities are compatible with that
scope and the active debug mode (see
:meth:`LoopFactorySnapshot.compatible`).

Negotiation results are published as immutable :class:`LoopFactorySnapshot`
objects. Publishing is atomic: if a factory provider fails to load or provides
an invalid capability, no new snapshot is published and snapshots that are
already effective stay untouched. The :class:`LoopFactoryNegotiator` is safe
to use from concurrent collection threads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from asyncio import AbstractEventLoop

LoopFactory: TypeAlias = Callable[[], AbstractEventLoop]

#: Loop scopes a factory can advertise support for.
NEGOTIABLE_SCOPES: frozenset[str] = frozenset(
    ("session", "package", "module", "class", "function")
)


class LoopFactoryNegotiationError(Exception):
    """Base class for failures while negotiating loop factory capabilities."""


class LoopFactoryLoadError(LoopFactoryNegotiationError):
    """
    Raised when a factory provider cannot be loaded.

    A load failure is isolated: the negotiation is discarded before it can be
    published, so a snapshot that is already effective is never polluted.
    """


class LoopFactoryDowngradeError(LoopFactoryNegotiationError):
    """
    Raised when an effective factory would be replaced by an older version.

    Effective snapshots only accept same-name factories with a higher version.
    """


class LoopFactoryConflictError(LoopFactoryNegotiationError):
    """
    Raised when two same-name, same-version factories disagree.

    Two factories share a name and version but provide a different callable,
    scope set, or debug capability. The already effective factory wins and the
    snapshot is left unchanged.
    """


@dataclass(frozen=True, slots=True)
class LoopFactoryCapability:
    """
    Immutable capability advertisement for a named loop factory.

    Args:
        name: Unique factory name. Also used as the parametrization test ID.
        factory: Zero-argument callable returning an event loop.
        scopes: Loop scopes supported by the factory. An empty collection
            means that every loop scope is supported.
        supports_debug: Whether the factory supports running with asyncio
            debug mode enabled.
        version: Monotonically increasing capability version. A same-name
            factory only overrides an effective factory when its version is
            higher.

    """

    name: str
    factory: LoopFactory
    scopes: frozenset[str] = NEGOTIABLE_SCOPES
    supports_debug: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise LoopFactoryLoadError(
                "Loop factory capabilities require a non-empty string name."
            )
        if not callable(self.factory):
            raise LoopFactoryLoadError(
                f"Loop factory {self.name!r} must be a callable."
            )
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise LoopFactoryLoadError(
                f"Loop factory {self.name!r} version must be an integer."
            )
        if self.version < 1:
            raise LoopFactoryLoadError(
                f"Loop factory {self.name!r} version must be >= 1."
            )
        scopes = frozenset(self.scopes)
        unsupported = scopes - NEGOTIABLE_SCOPES
        if unsupported:
            invalid_scopes = ", ".join(sorted(unsupported))
            raise LoopFactoryLoadError(
                f"Loop factory {self.name!r} advertises unsupported loop "
                f"scope(s): {invalid_scopes}."
            )
        if not scopes:
            # An empty scope set means "supports every scope".
            scopes = NEGOTIABLE_SCOPES
        object.__setattr__(self, "scopes", scopes)

    def supports_scope(self, scope: str) -> bool:
        """Return whether the factory supports the given loop scope."""
        return scope in self.scopes

    def compatible_with(self, scope: str, debug: bool = False) -> bool:
        """
        Return whether the factory can run a node in the given scope.

        The factory must support the scope and, when asyncio debug mode is
        requested, it must advertise debug support.
        """
        return self.supports_scope(scope) and (not debug or self.supports_debug)


def as_loop_factory_capability(
    name: str,
    value: LoopFactory | LoopFactoryCapability,
) -> LoopFactoryCapability:
    """
    Coerce a hook mapping value into a loop factory capability.

    A bare callable is treated as a factory that supports every scope and
    asyncio debug mode at version 1, preserving backward compatibility with
    hooks that return plain ``name -> callable`` mappings.
    """
    if isinstance(value, LoopFactoryCapability):
        if value.name != name:
            raise LoopFactoryLoadError(
                f"Loop factory capability is named {value.name!r} but was "
                f"registered under {name!r}."
            )
        return value
    if callable(value):
        return LoopFactoryCapability(name=name, factory=value)
    raise LoopFactoryLoadError(
        f"Loop factory {name!r} must be a callable or a LoopFactoryCapability."
    )


class LoopFactorySnapshot:
    """
    Immutable result of negotiating loop factory capabilities.

    A snapshot maps factory names to their negotiated capabilities. Instances
    cannot be modified after creation; negotiating new capabilities publishes
    a new snapshot instead of mutating an effective one.
    """

    __slots__ = ("_factories", "_key")

    def __init__(
        self,
        key: Hashable,
        factories: Iterable[LoopFactoryCapability],
    ) -> None:
        entries: dict[str, LoopFactoryCapability] = {}
        for capability in factories:
            if not isinstance(capability, LoopFactoryCapability):
                raise LoopFactoryLoadError(
                    "Loop factory snapshots can only contain "
                    "LoopFactoryCapability objects."
                )
            if capability.name in entries:
                raise LoopFactoryLoadError(
                    f"Duplicate loop factory name {capability.name!r} in a "
                    "single negotiation snapshot."
                )
            entries[capability.name] = capability
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_factories", MappingProxyType(entries))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("LoopFactorySnapshot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("LoopFactorySnapshot is immutable")

    @property
    def key(self) -> Hashable:
        """Identity of the scope boundary this snapshot belongs to."""
        return self._key

    @property
    def factories(self) -> Mapping[str, LoopFactoryCapability]:
        """Read-only mapping of factory names to negotiated capabilities."""
        return self._factories

    @property
    def names(self) -> tuple[str, ...]:
        """Negotiated factory names in publication order."""
        return tuple(self._factories)

    def compatible(
        self,
        scope: str,
        *,
        debug: bool = False,
    ) -> tuple[LoopFactoryCapability, ...]:
        """
        Return the factories a node entering *scope* may accept.

        Factories that do not support the scope or the requested debug mode
        are excluded, so callers only see compatible factories.
        """
        return tuple(
            capability
            for capability in self._factories.values()
            if capability.compatible_with(scope, debug)
        )

    def __getitem__(self, name: str) -> LoopFactoryCapability:
        return self._factories[name]

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self._factories)

    def __len__(self) -> int:
        return len(self._factories)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LoopFactorySnapshot):
            return NotImplemented
        if self._key != other._key:
            return False
        return tuple(sorted(self._factories.items())) == tuple(
            sorted(other._factories.items())
        )

    def __hash__(self) -> int:
        return hash((self._key, tuple(sorted(self._factories.items()))))

    def __repr__(self) -> str:
        names = ", ".join(self._factories)
        return f"LoopFactorySnapshot(key={self._key!r}, factories=[{names}])"


#: A provider returns the named factories available for a collected node.
#: It may be passed directly as a mapping or as a zero-argument loader so
#: that loader failures can be contained by the negotiator.
LoopFactoryProvider: TypeAlias = (
    Callable[[], Mapping[str, LoopFactory | LoopFactoryCapability]]
    | Mapping[str, LoopFactory | LoopFactoryCapability]
)


class LoopFactoryNegotiator:
    """
    Negotiates and publishes immutable loop factory snapshots.

    One negotiator is shared by a pytest session. Snapshots are keyed by the
    loop scope boundary a node enters. All operations are serialized with a
    reentrant lock, so concurrent collection threads can negotiate safely.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshots: dict[Hashable, LoopFactorySnapshot] = {}

    def effective_snapshot(
        self,
        key: Hashable,
    ) -> LoopFactorySnapshot | None:
        """Return the snapshot currently effective for *key*, if any."""
        with self._lock:
            return self._snapshots.get(key)

    def effective_keys(self) -> tuple[Hashable, ...]:
        """Return the keys for which a snapshot has been published."""
        with self._lock:
            return tuple(self._snapshots)

    def negotiate(
        self,
        key: Hashable,
        providers: Iterable[LoopFactoryProvider],
    ) -> LoopFactorySnapshot:
        """
        Load *providers* and publish an atomic snapshot for *key*.

        Providers are loaded into a private candidate. A provider that raises,
        an invalid capability, a same-name version downgrade, or a same-name
        conflict discards the whole candidate without touching the effective
        snapshot. A same-name factory with a higher version overrides the
        effective factory; re-publishing an identical capability is a no-op.
        """
        candidate: list[LoopFactoryCapability] = []
        with self._lock:
            for provider in providers:
                mapping = self._load_provider(key, provider)
                for name, value in mapping.items():
                    capability = as_loop_factory_capability(name, value)
                    if any(existing.name == capability.name for existing in candidate):
                        raise LoopFactoryLoadError(
                            f"Duplicate loop factory name {capability.name!r} "
                            "while negotiating a single snapshot."
                        )
                    candidate.append(capability)

            current = self._snapshots.get(key)
            merged: dict[str, LoopFactoryCapability] = (
                dict(current.factories) if current is not None else {}
            )
            for capability in candidate:
                effective = merged.get(capability.name)
                if effective is None:
                    merged[capability.name] = capability
                elif capability.version > effective.version:
                    merged[capability.name] = capability
                elif capability == effective:
                    # Identical re-negotiation, e.g. a parametrized node or a
                    # concurrent collector loading the same factories.
                    continue
                elif capability.version < effective.version:
                    raise LoopFactoryDowngradeError(
                        f"Refusing to replace loop factory {capability.name!r} "
                        f"version {effective.version} with older version "
                        f"{capability.version}."
                    )
                else:
                    raise LoopFactoryConflictError(
                        f"Loop factory {capability.name!r} version "
                        f"{capability.version} is already effective with "
                        "different capabilities."
                    )
            # Construct before publishing: a constructor failure must not
            # replace the effective snapshot.
            snapshot = LoopFactorySnapshot(key, merged.values())
            self._snapshots[key] = snapshot
            return snapshot

    @staticmethod
    def _load_provider(
        key: Hashable,
        provider: LoopFactoryProvider,
    ) -> Mapping[str, LoopFactory | LoopFactoryCapability]:
        try:
            if callable(provider):
                result = provider()
            else:
                result = provider
        except LoopFactoryNegotiationError:
            raise
        except Exception as error:
            raise LoopFactoryLoadError(
                f"Failed to load loop factories for snapshot key {key!r}: {error}"
            ) from error
        if not isinstance(result, Mapping):
            raise LoopFactoryLoadError(
                f"Loop factory provider for snapshot key {key!r} must return a mapping."
            )
        return result

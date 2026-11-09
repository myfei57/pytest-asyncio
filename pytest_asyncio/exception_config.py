"""
Scope-aware merging and snapshot/restore of event loop exception configuration.

An asyncio event loop exposes three knobs that tests may want to configure:

* the custom exception handler (``loop.set_exception_handler``),
* the asyncio debug mode (``loop.set_debug``) and
* the slow callback duration (``loop.slow_callback_duration``).

Configuration can be provided at every pytest scope, from ``session`` down to
``function``.  :class:`ExceptionHandlerConfigMerger` merges those layers and
records both the source scope of each effective value and the full shadowing
chain (which outer layer was overridden by which inner layer).

:func:`loop_exception_config` guards a section of test execution: it takes an
immutable snapshot of the loop before entering the section and restores it
atomically when leaving.  Unknown handlers and restore conflicts never mask
the exception raised by the test itself -- they are chained onto it and carry
enough location information to pinpoint the offending scope and node.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pytest_asyncio.plugin import PytestAsyncioError


class _Unset(enum.Enum):
    """Sentinel for values that no configuration layer provides."""

    UNSET = 0


#: Sentinel used when a setting is not configured at a given layer.  It is
#: distinct from ``None``, which is a valid exception handler value.
UNSET = _Unset.UNSET


#: Pytest scopes ordered from the outermost to the innermost layer.
SCOPE_ORDER: tuple[str, ...] = (
    "session",
    "package",
    "module",
    "class",
    "function",
)

#: Names of the event loop settings managed by this module.
CONFIG_FIELDS: tuple[str, ...] = (
    "exception_handler",
    "debug",
    "slow_callback_duration",
)

#: Signature of an asyncio event loop exception handler.  Mirrors the
#: handler type used by the standard library's event loop APIs.
ExceptionHandler = Callable[
    [asyncio.AbstractEventLoop, dict[str, Any]], object
]


class ExceptionConfigError(PytestAsyncioError):
    """Base class for errors related to exception handler configuration."""


class UnknownExceptionHandlerError(ExceptionConfigError):
    """Raised when a configured handler is neither callable nor ``None``."""


class ExceptionConfigRestoreError(ExceptionConfigError):
    """Raised when a loop cannot be restored to its snapshotted state."""


def _location(scope: str | None, nodeid: str) -> str:
    if scope is None:
        return f"nodeid={nodeid!r}"
    return f"scope={scope!r}, nodeid={nodeid!r}"


@dataclass(frozen=True)
class ExceptionHandlerSettings:
    """
    Exception handling configuration provided at a single scope.

    Fields left at :data:`UNSET` are not configured at this layer and do not
    participate in merging.
    """

    exception_handler: ExceptionHandler | _Unset | None = UNSET
    debug: bool | _Unset = UNSET
    slow_callback_duration: float | _Unset = UNSET


@dataclass(frozen=True)
class ScopedSettings:
    """A settings layer attached to a particular pytest scope."""

    scope: str
    nodeid: str
    settings: ExceptionHandlerSettings


@dataclass(frozen=True)
class FieldOrigin:
    """The layer that provides the effective value of a single field."""

    scope: str
    nodeid: str
    value: Any


@dataclass(frozen=True)
class ShadowEntry:
    """
    One entry in the shadowing chain of a single field.

    ``shadowed_by`` is the scope of the inner layer that overrides this entry,
    or ``None`` if this entry itself provides the effective value.
    """

    scope: str
    nodeid: str
    value: Any
    shadowed_by: str | None


@dataclass(frozen=True)
class MergedExceptionHandlerConfig:
    """
    Result of merging every configuration layer.

    A field left at :data:`UNSET` means that no layer configured it.
    ``sources`` maps each field to the layer that won the merge and
    ``shadow_chains`` maps it to every layer that touched it, ordered from
    outer to inner scope.
    """

    exception_handler: Any
    debug: Any
    slow_callback_duration: Any
    sources: Mapping[str, FieldOrigin | None]
    shadow_chains: Mapping[str, tuple[ShadowEntry, ...]]


class ExceptionHandlerConfigMerger:
    """Merge exception handler settings from outer scopes into inner scopes."""

    def __init__(self, layers: Sequence[ScopedSettings]) -> None:
        for layer in layers:
            if layer.scope not in SCOPE_ORDER:
                raise ExceptionConfigError(
                    f"Unknown configuration scope {layer.scope!r} at "
                    f"nodeid={layer.nodeid!r}. Valid scopes: "
                    f"{', '.join(SCOPE_ORDER)}."
                )
        # Stable sort keeps nested layers within the same scope (e.g. nested
        # packages) in their original outer-to-inner order.
        self._layers = sorted(layers, key=lambda layer: SCOPE_ORDER.index(layer.scope))

    @property
    def layers(self) -> tuple[ScopedSettings, ...]:
        return tuple(self._layers)

    def merge(self) -> MergedExceptionHandlerConfig:
        """
        Merge all layers, validating every configured value.

        Raises :class:`ExceptionConfigError` (or one of its subclasses) if a
        layer carries an invalid value.  The error message locates the
        offending scope and node.
        """
        self._validate()
        effective: dict[str, Any] = {}
        sources: dict[str, FieldOrigin | None] = {}
        chains: dict[str, tuple[ShadowEntry, ...]] = {}
        for field_name in CONFIG_FIELDS:
            entries = [
                (layer, getattr(layer.settings, field_name))
                for layer in self._layers
                if getattr(layer.settings, field_name) is not UNSET
            ]
            chain = tuple(
                ShadowEntry(
                    scope=layer.scope,
                    nodeid=layer.nodeid,
                    value=value,
                    shadowed_by=(
                        entries[index + 1][0].scope
                        if index + 1 < len(entries)
                        else None
                    ),
                )
                for index, (layer, value) in enumerate(entries)
            )
            chains[field_name] = chain
            if entries:
                winner_layer, winner_value = entries[-1]
                effective[field_name] = winner_value
                sources[field_name] = FieldOrigin(
                    scope=winner_layer.scope,
                    nodeid=winner_layer.nodeid,
                    value=winner_value,
                )
            else:
                effective[field_name] = UNSET
                sources[field_name] = None
        return MergedExceptionHandlerConfig(
            exception_handler=effective["exception_handler"],
            debug=effective["debug"],
            slow_callback_duration=effective["slow_callback_duration"],
            sources=sources,
            shadow_chains=chains,
        )

    def _validate(self) -> None:
        for layer in self._layers:
            settings = layer.settings
            handler = settings.exception_handler
            if handler is not UNSET and handler is not None and not callable(handler):
                raise UnknownExceptionHandlerError(
                    f"Unknown exception handler {handler!r} configured at "
                    f"{_location(layer.scope, layer.nodeid)}. An exception "
                    "handler must be a callable or None."
                )
            debug = settings.debug
            if debug is not UNSET and not isinstance(debug, bool):
                raise ExceptionConfigError(
                    f"Invalid debug value {debug!r} at "
                    f"{_location(layer.scope, layer.nodeid)}: expected a "
                    "boolean."
                )
            duration = settings.slow_callback_duration
            if duration is not UNSET and (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or duration < 0
            ):
                raise ExceptionConfigError(
                    "Invalid slow_callback_duration value "
                    f"{duration!r} at {_location(layer.scope, layer.nodeid)}: "
                    "expected a non-negative number."
                )


@dataclass(frozen=True)
class LoopConfigSnapshot:
    """
    Immutable picture of an event loop's exception handling state.

    Instances cannot be mutated after creation, so the restore target stays
    trustworthy even if the loop itself is reconfigured later.
    """

    nodeid: str
    loop_id: int
    loop_repr: str
    exception_handler: ExceptionHandler | None
    debug: bool
    slow_callback_duration: float


@dataclass(frozen=True)
class _AppliedLoopConfig:
    """State the merger actually applied, used for conflict detection."""

    exception_handler: ExceptionHandler | None
    debug: bool
    slow_callback_duration: float


def snapshot_loop_config(
    loop: asyncio.AbstractEventLoop, nodeid: str
) -> LoopConfigSnapshot:
    """Return an immutable snapshot of the loop's exception handling state."""
    if loop.is_closed():
        raise ExceptionConfigError(
            f"Cannot snapshot a closed event loop at nodeid={nodeid!r} (loop={loop!r})."
        )
    return LoopConfigSnapshot(
        nodeid=nodeid,
        loop_id=id(loop),
        loop_repr=repr(loop),
        exception_handler=loop.get_exception_handler(),
        debug=loop.get_debug(),
        slow_callback_duration=loop.slow_callback_duration,
    )


def apply_loop_config(
    loop: asyncio.AbstractEventLoop,
    merged: MergedExceptionHandlerConfig,
    nodeid: str,
) -> _AppliedLoopConfig:
    """
    Apply a merged configuration to the loop.

    Every value is validated before the loop is touched, so an invalid
    configuration can never leave the loop partially updated.
    """
    if loop.is_closed():
        raise ExceptionConfigError(
            f"Cannot apply exception configuration at nodeid={nodeid!r}: "
            f"the event loop {loop!r} is closed."
        )
    handler = merged.exception_handler
    if handler is not UNSET and handler is not None and not callable(handler):
        # Defense in depth: the merger already rejects unknown handlers.
        raise UnknownExceptionHandlerError(
            f"Unknown exception handler {handler!r} cannot be applied at "
            f"nodeid={nodeid!r}: expected a callable or None."
        )
    current_handler = loop.get_exception_handler()
    current_debug = loop.get_debug()
    current_duration = loop.slow_callback_duration
    applied = _AppliedLoopConfig(
        exception_handler=current_handler if handler is UNSET else handler,
        debug=current_debug if merged.debug is UNSET else merged.debug,
        slow_callback_duration=(
            current_duration
            if merged.slow_callback_duration is UNSET
            else float(merged.slow_callback_duration)
        ),
    )
    # Mutation starts only after every value above validated successfully.
    if handler is not UNSET:
        loop.set_exception_handler(handler)
    if merged.debug is not UNSET:
        loop.set_debug(merged.debug)
    if merged.slow_callback_duration is not UNSET:
        loop.slow_callback_duration = float(merged.slow_callback_duration)
    return applied


def restore_loop_config(
    loop: asyncio.AbstractEventLoop,
    snapshot: LoopConfigSnapshot,
    applied: _AppliedLoopConfig,
) -> None:
    """
    Restore the loop to its snapshotted state.

    The restore is atomic in the sense that every field is restored even when
    one or more fields conflict with state changed outside of the
    configuration layer.  All conflicts are collected and reported in a
    single :class:`ExceptionConfigRestoreError` that locates the affected
    node and loop.
    """
    nodeid = snapshot.nodeid
    if id(loop) != snapshot.loop_id:
        raise ExceptionConfigRestoreError(
            "Exception configuration restore conflict at "
            f"nodeid={nodeid!r}: the snapshot was taken from loop "
            f"{snapshot.loop_repr} (id={snapshot.loop_id}) but a different "
            f"loop {loop!r} (id={id(loop)}) was passed for restoration."
        )
    if loop.is_closed():
        raise ExceptionConfigRestoreError(
            "Exception configuration restore conflict at "
            f"nodeid={nodeid!r}: the loop {snapshot.loop_repr} was closed "
            "before its configuration could be restored."
        )

    conflicts = []
    if loop.get_exception_handler() is not applied.exception_handler:
        conflicts.append(
            "exception_handler was changed outside of the configuration "
            f"layer: expected {applied.exception_handler!r}, found "
            f"{loop.get_exception_handler()!r}"
        )
    if loop.get_debug() != applied.debug:
        conflicts.append(
            "debug was changed outside of the configuration layer: "
            f"expected {applied.debug!r}, found {loop.get_debug()!r}"
        )
    if loop.slow_callback_duration != applied.slow_callback_duration:
        conflicts.append(
            "slow_callback_duration was changed outside of the configuration "
            f"layer: expected {applied.slow_callback_duration!r}, found "
            f"{loop.slow_callback_duration!r}"
        )

    failures = []
    restore_steps = (
        (
            "exception_handler",
            lambda: loop.set_exception_handler(snapshot.exception_handler),
        ),
        ("debug", lambda: loop.set_debug(snapshot.debug)),
        (
            "slow_callback_duration",
            lambda: setattr(
                loop, "slow_callback_duration", snapshot.slow_callback_duration
            ),
        ),
    )
    for field_name, restore_step in restore_steps:
        try:
            restore_step()
        except Exception as exc:
            failures.append(
                f"failed to restore {field_name} at nodeid={nodeid!r}: {exc!r}"
            )

    problems = conflicts + failures
    if problems:
        details = "\n  ".join(problems)
        raise ExceptionConfigRestoreError(
            "Exception configuration restore conflict at "
            f"nodeid={nodeid!r} on loop {snapshot.loop_repr} "
            f"(id={snapshot.loop_id}):\n  {details}"
        )


@contextlib.contextmanager
def loop_exception_config(
    loop: asyncio.AbstractEventLoop,
    merged: MergedExceptionHandlerConfig,
    nodeid: str,
) -> Iterator[LoopConfigSnapshot]:
    """
    Snapshot, apply and atomically restore loop exception configuration.

    If the guarded section raises and the restore itself fails, the restore
    error is raised with the original exception attached as its cause, so the
    main test/setup exception is never lost.
    """
    snapshot = snapshot_loop_config(loop, nodeid)
    applied = apply_loop_config(loop, merged, nodeid)
    try:
        yield snapshot
    except BaseException as main_exception:
        try:
            restore_loop_config(loop, snapshot, applied)
        except ExceptionConfigError as restore_error:
            raise restore_error from main_exception
        raise
    else:
        restore_loop_config(loop, snapshot, applied)


__all__ = [
    "CONFIG_FIELDS",
    "SCOPE_ORDER",
    "UNSET",
    "ExceptionConfigError",
    "ExceptionConfigRestoreError",
    "ExceptionHandler",
    "ExceptionHandlerConfigMerger",
    "ExceptionHandlerSettings",
    "FieldOrigin",
    "LoopConfigSnapshot",
    "MergedExceptionHandlerConfig",
    "ScopedSettings",
    "ShadowEntry",
    "UnknownExceptionHandlerError",
    "apply_loop_config",
    "loop_exception_config",
    "restore_loop_config",
    "snapshot_loop_config",
]

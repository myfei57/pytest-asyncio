"""
Ordering barrier for asynchronous test completion events.

Each asynchronous test runs through three lifecycle phases that must be
observed in a fixed order: the test call, fixture teardown and the closing
of the event loop. Completion callbacks do not necessarily arrive in that
order -- the loop of a function-scoped runner, for example, is closed while
fixture teardown is still in progress, so its callback arrives early.

The barrier registers the expected events up front. Every registration
returns a monotonically increasing sequence number. Callbacks submit their
events through :meth:`CompletionBarrier.ingest`; out-of-order events are
held back in a staging area instead of being forwarded immediately. When
the barrier is closed, all staged events are published to the subscriber
ordered by phase and, within a phase, by sequence number.

Events belonging to another test, duplicate events and events submitted
after the barrier was closed are rejected with a :class:`BarrierError`.
The barrier is purely observational: it never touches the exception or the
outcome of the test it observes.
"""

from __future__ import annotations

import enum
import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class CompletionPhase(enum.IntEnum):
    """Lifecycle phases of an asynchronous test, in publication order."""

    TEST = 1
    FIXTURE_TEARDOWN = 2
    LOOP_CLOSE = 3


class BarrierError(Exception):
    """Base class for every error raised by the ordering barrier."""


class UnknownEventError(BarrierError):
    """An event was ingested whose sequence number was never registered."""


class DuplicateEventError(BarrierError):
    """An event with the same sequence number was ingested more than once."""


class CrossTestEventError(BarrierError):
    """An event bound to another test was ingested into this barrier."""


class PhaseClosedError(BarrierError):
    """An event was registered or ingested after its phase was closed."""


class BarrierClosedError(BarrierError):
    """The barrier was used after it had already been closed."""


class DuplicateCloseError(BarrierError):
    """A phase, or the barrier itself, was closed more than once."""


@dataclass(frozen=True)
class CompletionEvent:
    """A single completion observation collected by the barrier."""

    test_id: str
    seq: int
    phase: CompletionPhase
    payload: Any = None


#: Receives every event when the barrier flushes its staging area.
EventSubscriber = Callable[[CompletionEvent], None]


class CompletionBarrier:
    """
    Collects completion events of one test and publishes them in order.

    The barrier is bound to a single ``test_id``; events carrying any other
    test id are rejected. All methods are thread-safe, so events may be
    ingested concurrently from several tasks or threads.
    """

    def __init__(
        self,
        test_id: str,
        subscriber: EventSubscriber | None = None,
    ) -> None:
        self._test_id = test_id
        self._subscriber = subscriber
        self._lock = threading.Lock()
        self._next_seq = 0
        self._phase_by_seq: dict[int, CompletionPhase] = {}
        self._staged: dict[int, CompletionEvent] = {}
        self._delivered: set[int] = set()
        self._closed_phases: set[CompletionPhase] = set()
        self._closed = False
        self.published: list[CompletionEvent] = []

    @property
    def test_id(self) -> str:
        """The nodeid of the test this barrier is bound to."""
        return self._test_id

    @property
    def closed(self) -> bool:
        """Whether the barrier has already been closed."""
        with self._lock:
            return self._closed

    @property
    def missing_events(self) -> list[int]:
        """Sequence numbers that were registered but never ingested."""
        with self._lock:
            return sorted(
                seq
                for seq in self._phase_by_seq
                if seq not in self._staged and seq not in self._delivered
            )

    def register(self, phase: CompletionPhase) -> int:
        """
        Register an expected event for ``phase`` and return its seq.

        Sequence numbers are handed out in strictly monotonic order across
        all phases of this barrier.
        """
        with self._lock:
            self._ensure_open()
            if phase in self._closed_phases:
                raise PhaseClosedError(
                    f"Cannot register an event for closed phase {phase.name!r}."
                )
            seq = self._next_seq
            self._next_seq += 1
            self._phase_by_seq[seq] = phase
            return seq

    def register_lifecycle(self) -> tuple[int, int, int]:
        """Register one milestone event for each of the three phases."""
        return (
            self.register(CompletionPhase.TEST),
            self.register(CompletionPhase.FIXTURE_TEARDOWN),
            self.register(CompletionPhase.LOOP_CLOSE),
        )

    def ingest(self, seq: int, test_id: str, payload: Any = None) -> None:
        """
        Submit a completion callback to the staging area.

        Raises:
            BarrierClosedError: if the barrier is already closed.
            CrossTestEventError: if ``test_id`` does not match the bound test.
            UnknownEventError: if ``seq`` was never registered.
            DuplicateEventError: if ``seq`` was already ingested before.
            PhaseClosedError: if the event's phase was closed already.

        """
        with self._lock:
            self._ensure_open()
            if test_id != self._test_id:
                raise CrossTestEventError(
                    f"Event for {test_id!r} was rejected by the barrier bound "
                    f"to {self._test_id!r}."
                )
            phase = self._phase_by_seq.get(seq)
            if phase is None:
                raise UnknownEventError(
                    f"Event with unregistered sequence number {seq} was rejected."
                )
            if seq in self._staged or seq in self._delivered:
                raise DuplicateEventError(
                    f"Duplicate event with sequence number {seq} was rejected."
                )
            if phase in self._closed_phases:
                raise PhaseClosedError(
                    f"Event {seq} arrived after phase {phase.name!r} was closed."
                )
            self._staged[seq] = CompletionEvent(test_id, seq, phase, payload)

    def close_phase(self, phase: CompletionPhase) -> None:
        """Mark ``phase`` as finished; later events for it are rejected."""
        with self._lock:
            self._ensure_open()
            if phase in self._closed_phases:
                raise DuplicateCloseError(
                    f"Phase {phase.name!r} was closed more than once."
                )
            self._closed_phases.add(phase)

    def close(self) -> list[CompletionEvent]:
        """
        Close the barrier and publish the staged events in phase order.

        Any phase not closed explicitly is closed implicitly. Events that
        were registered but never ingested stay accessible through
        :attr:`missing_events` and do not prevent the close, so a failed
        teardown cannot block the remaining events from being published.
        Closing twice raises :class:`DuplicateCloseError` and never
        republishes any event.
        """
        with self._lock:
            if self._closed:
                raise DuplicateCloseError(
                    f"Barrier for {self._test_id!r} was closed more than once."
                )
            self._closed = True
            ordered = sorted(
                self._staged.values(),
                key=lambda event: (event.phase, event.seq),
            )
            for event in ordered:
                self._delivered.add(event.seq)
                self.published.append(event)
            self._staged.clear()
            subscriber = self._subscriber
        # Invoke the subscriber outside the lock so that subscribers are
        # free to acquire their own locks without risking a deadlock.
        for event in ordered:
            if subscriber is not None:
                subscriber(event)
        return list(self.published)

    def _ensure_open(self) -> None:
        if self._closed:
            raise BarrierClosedError(
                f"Barrier for {self._test_id!r} is already closed."
            )


class CompletionRegistry:
    """
    Holds the completion barriers of all tests of one pytest session.

    The registry also tracks which tests share a given asyncio runner, so
    that the loop-close phase of a wider-scoped runner can close the
    barriers of several tests at once. Every mutating helper is meant to be
    called through :meth:`safe`, which guarantees that a barrier violation
    is recorded but never escapes into the pytest lifecycle.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._barriers: dict[str, CompletionBarrier] = {}
        self._milestones: dict[str, tuple[int, int, int]] = {}
        self._runner_tests: dict[int, list[str]] = {}
        self._closed_runners: set[int] = set()
        self._finished_tests: set[str] = set()
        self.events: dict[str, list[CompletionEvent]] = {}
        self.errors: list[str] = []

    def begin_test(self, test_id: str) -> CompletionBarrier:
        """Create the barrier of a test and register its three milestones."""
        with self._lock:
            existing = self._barriers.get(test_id)
            if existing is not None:
                # A protocol must not run twice, but stay defensive so the
                # barrier can never alter the outcome of a test.
                return existing
            barrier = CompletionBarrier(test_id, subscriber=self._publish)
            self._milestones[test_id] = barrier.register_lifecycle()
            self._barriers[test_id] = barrier
            self.events[test_id] = []
            return barrier

    def bind_runner(self, test_id: str, runner: object) -> None:
        """Remember that ``test_id`` runs its event loop on ``runner``."""
        with self._lock:
            self._runner_tests.setdefault(id(runner), []).append(test_id)

    def complete_test_phase(self, test_id: str) -> None:
        """Record the completion of the test-call phase."""
        self._ingest_milestone(test_id, 0, CompletionPhase.TEST.name)

    def complete_teardown_phase(self, test_id: str) -> None:
        """Record the completion of the fixture-teardown phase."""
        self._ingest_milestone(test_id, 1, CompletionPhase.FIXTURE_TEARDOWN.name)

    def complete_loop_close(self, runner: object) -> None:
        """
        Record the loop-close milestone of every test bound to ``runner``.

        Barriers whose test protocol has already finished are closed right
        away; the last consumer of a wider-scoped runner closes its own
        barrier from :meth:`finish_test`.
        """
        with self._lock:
            test_ids = self._runner_tests.pop(id(runner), [])
            self._closed_runners.add(id(runner))
        for test_id in test_ids:
            self.safe(functools.partial(self._record_loop_close, test_id))

    def finish_test(self, test_id: str) -> None:
        """
        Safety-net close at the end of a test protocol.

        A barrier bound to a still-running wider-scoped runner stays open
        until that runner is torn down. Every other barrier is closed here,
        even if some milestones are missing.
        """
        with self._lock:
            barrier = self._barriers.get(test_id)
            if barrier is None or barrier.closed:
                self._finished_tests.discard(test_id)
                return
            self._finished_tests.add(test_id)
            runner_still_alive = any(
                test_id in test_ids and runner_id not in self._closed_runners
                for runner_id, test_ids in self._runner_tests.items()
            )
        if runner_still_alive:
            return
        barrier.close()

    def safe(self, action: Callable[[], object]) -> None:
        """Run ``action`` without ever letting a barrier error escape."""
        try:
            action()
        except BarrierError as exc:
            with self._lock:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def get_barrier(self, test_id: str) -> CompletionBarrier | None:
        """Return the barrier of ``test_id`` if one exists."""
        with self._lock:
            return self._barriers.get(test_id)

    def tracks(self, test_id: str) -> bool:
        """Whether a barrier exists for ``test_id``."""
        with self._lock:
            return test_id in self._barriers

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the recorded state."""
        with self._lock:
            return {
                "events": {
                    test_id: [
                        {
                            "seq": event.seq,
                            "phase": event.phase.name,
                            "payload": event.payload,
                        }
                        for event in events
                    ]
                    for test_id, events in self.events.items()
                },
                "missing": {
                    test_id: barrier.missing_events
                    for test_id, barrier in self._barriers.items()
                },
                "closed": {
                    test_id: barrier.closed
                    for test_id, barrier in self._barriers.items()
                },
                "errors": list(self.errors),
            }

    def _ingest_milestone(
        self,
        test_id: str,
        index: int,
        payload: str,
    ) -> None:
        with self._lock:
            barrier = self._barriers.get(test_id)
            milestones = self._milestones.get(test_id)
        if barrier is None or milestones is None:
            return
        barrier.ingest(milestones[index], test_id, payload=payload)

    def _record_loop_close(self, test_id: str) -> None:
        with self._lock:
            barrier = self._barriers.get(test_id)
            milestones = self._milestones.get(test_id)
            protocol_finished = test_id in self._finished_tests
        if barrier is None or milestones is None:
            return
        barrier.ingest(
            milestones[2],
            test_id,
            payload=CompletionPhase.LOOP_CLOSE.name,
        )
        if protocol_finished:
            barrier.close()

    def _publish(self, event: CompletionEvent) -> None:
        with self._lock:
            self.events.setdefault(event.test_id, []).append(event)

"""Unit tests for the asynchronous test completion ordering barrier."""

from __future__ import annotations

import random
import threading

import pytest

from pytest_asyncio.completion_barrier import (
    BarrierClosedError,
    CompletionBarrier,
    CompletionEvent,
    CompletionPhase,
    CompletionRegistry,
    CrossTestEventError,
    DuplicateCloseError,
    DuplicateEventError,
    PhaseClosedError,
    UnknownEventError,
)

PHASES = list(CompletionPhase)


def test_register_hands_out_monotonic_sequence_numbers_across_phases():
    barrier = CompletionBarrier("test_monotonic")
    seqs = [barrier.register(phase) for phase in PHASES for _ in range(2)]
    assert seqs == list(range(6))


def test_lifecycle_registration_covers_all_three_phases_in_order():
    barrier = CompletionBarrier("test_lifecycle")
    seqs = barrier.register_lifecycle()
    assert seqs == (0, 1, 2)
    for seq in seqs:
        barrier.ingest(seq, "test_lifecycle")
    published = barrier.close()
    assert [event.phase for event in published] == PHASES


def test_out_of_order_callbacks_are_staged_and_flushed_in_phase_order():
    barrier = CompletionBarrier("test_reorder")
    test_seq, teardown_seq, loop_seq = barrier.register_lifecycle()

    # The loop-close callback of a function-scoped runner can arrive before
    # the fixture-teardown milestone.
    barrier.ingest(loop_seq, "test_reorder", payload="loop")
    barrier.ingest(test_seq, "test_reorder", payload="test")
    assert barrier.published == []

    barrier.ingest(teardown_seq, "test_reorder", payload="teardown")
    assert barrier.published == []

    published = barrier.close()
    assert [event.seq for event in published] == [test_seq, teardown_seq, loop_seq]
    assert [event.phase for event in published] == PHASES
    assert [event.payload for event in published] == [
        "test",
        "teardown",
        "loop",
    ]


def test_events_within_a_phase_keep_their_registration_order():
    barrier = CompletionBarrier("test_phase_internal_order")
    first = barrier.register(CompletionPhase.TEST)
    second = barrier.register(CompletionPhase.TEST)
    teardown = barrier.register(CompletionPhase.FIXTURE_TEARDOWN)

    barrier.ingest(second, "test_phase_internal_order")
    barrier.ingest(teardown, "test_phase_internal_order")
    barrier.ingest(first, "test_phase_internal_order")

    published = barrier.close()
    assert [event.seq for event in published] == [first, second, teardown]


def test_subscriber_receives_the_published_events_in_order():
    delivered: list[CompletionEvent] = []
    barrier = CompletionBarrier("test_subscriber", subscriber=delivered.append)
    test_seq, teardown_seq, loop_seq = barrier.register_lifecycle()
    barrier.ingest(loop_seq, "test_subscriber")
    barrier.ingest(teardown_seq, "test_subscriber")
    barrier.ingest(test_seq, "test_subscriber")

    barrier.close()

    assert [event.seq for event in delivered] == [
        test_seq,
        teardown_seq,
        loop_seq,
    ]


def test_duplicate_event_is_rejected_and_not_published_twice():
    barrier = CompletionBarrier("test_duplicate")
    seq = barrier.register(CompletionPhase.TEST)
    barrier.ingest(seq, "test_duplicate")

    with pytest.raises(DuplicateEventError):
        barrier.ingest(seq, "test_duplicate")

    published = barrier.close()
    assert [event.seq for event in published] == [seq]


def test_event_with_unregistered_sequence_is_rejected():
    barrier = CompletionBarrier("test_unknown")
    with pytest.raises(UnknownEventError):
        barrier.ingest(42, "test_unknown")
    assert barrier.close() == []


def test_event_from_another_test_is_rejected():
    barrier = CompletionBarrier("test_owner")
    seq = barrier.register(CompletionPhase.TEST)

    with pytest.raises(CrossTestEventError):
        barrier.ingest(seq, "test_intruder")

    # The rejected callback must not have reached the staging area.
    assert barrier.close() == []


def test_closed_phase_rejects_late_registration_and_late_events():
    barrier = CompletionBarrier("test_phase_closed")
    seq = barrier.register(CompletionPhase.TEST)
    barrier.close_phase(CompletionPhase.TEST)

    with pytest.raises(PhaseClosedError):
        barrier.register(CompletionPhase.TEST)
    with pytest.raises(PhaseClosedError):
        barrier.ingest(seq, "test_phase_closed")
    with pytest.raises(DuplicateCloseError):
        barrier.close_phase(CompletionPhase.TEST)

    # Other phases remain usable.
    teardown_seq = barrier.register(CompletionPhase.FIXTURE_TEARDOWN)
    barrier.ingest(teardown_seq, "test_phase_closed")
    assert [event.seq for event in barrier.close()] == [teardown_seq]


def test_duplicate_barrier_close_is_rejected_and_does_not_republish():
    delivered: list[object] = []
    barrier = CompletionBarrier("test_double_close", subscriber=delivered.append)
    seq = barrier.register(CompletionPhase.TEST)
    barrier.ingest(seq, "test_double_close")

    barrier.close()
    assert barrier.closed

    with pytest.raises(DuplicateCloseError):
        barrier.close()

    assert len(delivered) == 1


def test_barrier_rejects_every_mutation_after_close():
    barrier = CompletionBarrier("test_usage_after_close")
    barrier.close()

    with pytest.raises(BarrierClosedError):
        barrier.register(CompletionPhase.TEST)
    with pytest.raises(BarrierClosedError):
        barrier.ingest(0, "test_usage_after_close")
    with pytest.raises(BarrierClosedError):
        barrier.close_phase(CompletionPhase.TEST)


def test_failed_teardown_does_not_block_close_and_leaves_missing_events():
    barrier = CompletionBarrier("test_teardown_failure")
    test_seq, teardown_seq, loop_seq = barrier.register_lifecycle()
    barrier.ingest(test_seq, "test_teardown_failure")
    # The teardown callback never arrives because the teardown raised.

    published = barrier.close()

    assert [event.seq for event in published] == [test_seq]
    assert barrier.missing_events == [teardown_seq, loop_seq]


def test_concurrent_callbacks_are_published_exactly_once_in_order():
    barrier = CompletionBarrier("test_concurrent")
    per_phase = {
        phase: [barrier.register(phase) for _ in range(10)] for phase in PHASES
    }
    submissions = [(phase, seq) for phase, seqs in per_phase.items() for seq in seqs]
    random.Random(20261108).shuffle(submissions)

    threads = [
        threading.Thread(
            target=barrier.ingest,
            args=(seq, "test_concurrent"),
            kwargs={"payload": phase.name},
        )
        for phase, seq in submissions
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    published = barrier.close()
    expected = sorted(submissions, key=lambda item: (item[0], item[1]))
    assert [(event.phase, event.seq) for event in published] == expected
    assert len({event.seq for event in published}) == len(published) == 30
    assert barrier.missing_events == []


def test_registry_keeps_barrier_open_until_the_shared_runner_closes():
    registry = CompletionRegistry()
    registry.begin_test("test_module_scoped")
    runner = object()
    registry.bind_runner("test_module_scoped", runner)
    registry.complete_test_phase("test_module_scoped")
    registry.complete_teardown_phase("test_module_scoped")

    # The protocol of the test ends, but the module-scoped loop is alive.
    registry.finish_test("test_module_scoped")
    barrier = registry.get_barrier("test_module_scoped")
    assert barrier is not None
    assert not barrier.closed

    registry.complete_loop_close(runner)

    assert barrier.closed
    events = registry.snapshot()["events"]["test_module_scoped"]
    assert [event["phase"] for event in events] == [
        "TEST",
        "FIXTURE_TEARDOWN",
        "LOOP_CLOSE",
    ]
    assert registry.snapshot()["errors"] == []


def test_registry_closes_unbound_barrier_with_missing_milestones():
    registry = CompletionRegistry()
    registry.begin_test("test_skipped")

    registry.finish_test("test_skipped")

    barrier = registry.get_barrier("test_skipped")
    assert barrier is not None and barrier.closed
    assert barrier.missing_events == [0, 1, 2]
    assert registry.snapshot()["errors"] == []


def test_registry_runner_close_closes_every_bound_test_barrier():
    registry = CompletionRegistry()
    runner = object()
    for index in range(3):
        test_id = f"test_shared_{index}"
        registry.begin_test(test_id)
        registry.bind_runner(test_id, runner)
        registry.complete_test_phase(test_id)
        registry.complete_teardown_phase(test_id)
        # Every test protocol ends before the shared runner is torn down.
        registry.finish_test(test_id)

    registry.complete_loop_close(runner)

    snapshot = registry.snapshot()
    assert all(snapshot["closed"].values())
    for test_id in snapshot["events"]:
        assert [event["phase"] for event in snapshot["events"][test_id]] == [
            "TEST",
            "FIXTURE_TEARDOWN",
            "LOOP_CLOSE",
        ]


def test_registry_loop_event_arriving_before_teardown_is_reordered():
    # Exact function-scoped lifecycle: the runner finalizer runs inside
    # pytest_runtest_teardown, i.e. before the teardown milestone is recorded.
    registry = CompletionRegistry()
    registry.begin_test("test_function_scoped")
    runner = object()
    registry.bind_runner("test_function_scoped", runner)
    registry.complete_test_phase("test_function_scoped")
    registry.complete_loop_close(runner)
    registry.complete_teardown_phase("test_function_scoped")
    registry.finish_test("test_function_scoped")

    snapshot = registry.snapshot()
    assert snapshot["errors"] == []
    assert snapshot["closed"] == {"test_function_scoped": True}
    assert [event["phase"] for event in snapshot["events"]["test_function_scoped"]] == [
        "TEST",
        "FIXTURE_TEARDOWN",
        "LOOP_CLOSE",
    ]


def test_registry_safe_records_barrier_errors_instead_of_raising():
    registry = CompletionRegistry()
    registry.begin_test("test_safe")
    barrier = registry.get_barrier("test_safe")
    assert barrier is not None
    barrier.close()

    # A duplicate close through the safety wrapper must not escape.
    registry.safe(barrier.close)

    snapshot = registry.snapshot()
    assert snapshot["errors"]
    assert all("DuplicateCloseError" in error for error in snapshot["errors"])

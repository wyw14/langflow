"""Unit tests for the static directed-graph cyclic grouping report.

These tests exercise ``cycle_group_report`` purely from node IDs and
source/target edges; no ``Graph`` or component is ever built or executed.
"""

import pytest
from lfx.graph.graph.cycles import (
    CycleGroupReport,
    CyclicGroup,
    InvalidEdge,
    cycle_group_report,
)


def test_empty_graph_returns_empty_report():
    report = cycle_group_report([], [])

    assert report == CycleGroupReport(groups=(), invalid_edges=())


def test_multiple_disjoint_cycles_are_reported_separately():
    nodes = ["a", "b", "c", "d", "e", "f"]
    edges = [
        ("a", "b"),
        ("b", "a"),
        ("c", "d"),
        ("d", "c"),
        ("e", "f"),  # acyclic chain, must be ignored
    ]

    report = cycle_group_report(nodes, edges)

    assert report.invalid_edges == ()
    assert report.groups
    assert report.groups == (
        CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),
        CyclicGroup(nodes=("c", "d"), edges=(("c", "d"), ("d", "c"))),
    )


def test_groups_nodes_and_edges_are_stably_sorted_by_id():
    # Edges and nodes are supplied out of order; the second cycle would be
    # discovered first by a graph walk, yet output must be ID ordered.
    nodes = ["z", "a", "y", "b"]
    edges = [("z", "y"), ("y", "z"), ("b", "a"), ("a", "b")]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (
        CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),
        CyclicGroup(nodes=("y", "z"), edges=(("y", "z"), ("z", "y"))),
    )


def test_cycles_sharing_a_node_merge_into_one_component():
    nodes = ["a", "b", "c"]
    # a <-> b and b <-> c share b, so all three are mutually reachable.
    edges = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (
        CyclicGroup(
            nodes=("a", "b", "c"),
            edges=(("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")),
        ),
    )


def test_one_way_linked_cycles_remain_separate_components():
    nodes = ["a", "b", "c", "d"]
    # a <-> b, a one-way bridge b -> c, then c <-> d. There is no path back
    # from {c, d} to {a, b}, so the bridge is internal to neither group.
    edges = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "d"), ("d", "c")]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (
        CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),
        CyclicGroup(nodes=("c", "d"), edges=(("c", "d"), ("d", "c"))),
    )


def test_self_loop_is_reported_as_a_single_node_group():
    nodes = ["x", "y", "z", "w"]
    edges = [
        ("x", "x"),  # self-loop
        ("y", "z"),
        ("z", "y"),
        # "w" is an isolated node without a self-loop and must be excluded.
    ]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (
        CyclicGroup(nodes=("x",), edges=(("x", "x"),)),
        CyclicGroup(nodes=("y", "z"), edges=(("y", "z"), ("z", "y"))),
    )


def test_self_loop_survives_external_incoming_edge():
    nodes = ["s", "x"]
    edges = [("s", "x"), ("x", "x")]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (CyclicGroup(nodes=("x",), edges=(("x", "x"),)),)


def test_acyclic_graph_has_no_groups():
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c"), ("a", "d")]

    report = cycle_group_report(nodes, edges)

    assert report.groups == ()
    assert report.invalid_edges == ()


def test_feeder_and_sink_nodes_and_external_edges_are_excluded():
    nodes = ["p", "a", "b", "q"]
    # p feeds the cycle and the cycle drains into q; neither is in an SCC.
    edges = [("p", "a"), ("a", "b"), ("b", "a"), ("b", "q")]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),)
    cyclic_nodes = {node for group in report.groups for node in group.nodes}
    assert cyclic_nodes == {"a", "b"}


def test_edges_referencing_unknown_nodes_report_positions_without_adding_nodes():
    nodes = {"a", "b"}
    edges = [
        ("a", "b"),  # 0: valid
        ("b", "missing_target"),  # 1: unknown target
        ("missing_source", "a"),  # 2: unknown source
        ("x", "y"),  # 3: both endpoints unknown
        ("z", "z"),  # 4: unknown self-loop, reported only once
        ("b", "a"),  # 5: valid, closes the a <-> b cycle
    ]

    report = cycle_group_report(nodes, edges)

    assert report.invalid_edges == (
        InvalidEdge(index=1, source="b", target="missing_target", missing=("missing_target",)),
        InvalidEdge(index=2, source="missing_source", target="a", missing=("missing_source",)),
        InvalidEdge(index=3, source="x", target="y", missing=("x", "y")),
        InvalidEdge(index=4, source="z", target="z", missing=("z",)),
    )
    assert report.groups == (CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),)
    # The rejected endpoints must never have been added to the analyzed graph.
    reported_nodes = {node for group in report.groups for node in group.nodes}
    assert reported_nodes == {"a", "b"}


def test_unknown_edge_cannot_create_a_phantom_cycle():
    nodes = {"a"}
    # The edge a -> ghost -> a looks cyclic, but "ghost" is not a known node,
    # so both edges are rejected by position and no cycle is inferred.
    edges = [("a", "ghost"), ("ghost", "a")]

    report = cycle_group_report(nodes, edges)

    assert report.groups == ()
    assert [invalid.index for invalid in report.invalid_edges] == [0, 1]


def test_mapping_edges_are_accepted():
    nodes = ["a", "b", "c"]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "a"},
    ]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),)


def test_duplicate_nodes_and_parallel_edges_are_deduplicated():
    nodes = ["a", "a", "b"]  # duplicate node ID
    edges = [
        ("a", "b"),
        ("b", "a"),
        ("a", "b"),  # duplicate pair
        ("b", "a"),  # duplicate pair
    ]

    report = cycle_group_report(nodes, edges)

    assert report.groups == (CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),)


def test_generators_are_supported_as_input():
    report = cycle_group_report(iter(["a", "b"]), iter([("a", "b"), ("b", "a")]))

    assert report.groups == (CyclicGroup(nodes=("a", "b"), edges=(("a", "b"), ("b", "a"))),)


def test_malformed_pair_raises_value_error_with_position():
    with pytest.raises(ValueError, match="position 0"):
        cycle_group_report(["a", "b", "c"], [("a", "b", "c")])


def test_mapping_missing_target_key_raises_value_error_with_position():
    with pytest.raises(ValueError, match=r"position 0.*target"):
        cycle_group_report(["a"], [{"source": "a"}])


def test_unsupported_edge_type_raises_type_error_with_position():
    with pytest.raises(TypeError, match="position 0"):
        cycle_group_report(["a"], [42])

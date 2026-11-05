"""Static cyclic grouping report for directed graphs.

This module groups the cyclic strongly connected components (SCCs) of a
directed graph given only node IDs and ``source``/``target`` edges. It is a
pure structural analysis: it never instantiates or executes
:class:`~lfx.graph.graph.base.Graph` and never runs a component.

A strongly connected component is reported only when it contains a cycle:

* an SCC with more than one node always contains a directed cycle;
* a single-node SCC is reported only when it has a self-loop edge.

Every edge that references a node that is not part of the provided node IDs
is rejected and reported by its position (index) in the input edge list
instead of implicitly adding the unknown node. Nodes within a group, edges
within a group, and the groups themselves are returned in stable ID order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import networkx as nx

__all__ = [
    "CycleGroupReport",
    "CyclicGroup",
    "InvalidEdge",
    "cycle_group_report",
]

# Number of endpoints in an edge pair.
EDGE_PAIR_LENGTH = 2

# An edge is a ``(source, target)`` pair (tuple or list) or a mapping holding
# ``"source"`` and ``"target"`` keys, mirroring the two edge shapes used across
# the codebase (edge tuples and serialized flow-edge dictionaries).
EdgeInput = Sequence[str] | Mapping[str, str]


@dataclass(frozen=True)
class CyclicGroup:
    """A strongly connected component that contains at least one directed cycle.

    Attributes:
        nodes: Node IDs of the component sorted by ID.
        edges: Distinct internal ``(source, target)`` edges whose endpoints are
            both in ``nodes``, sorted by ``(source, target)``.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class InvalidEdge:
    """An input edge that references one or two unknown node IDs.

    Attributes:
        index: Zero-based position of the edge in the input edge list.
        source: Source endpoint as provided.
        target: Target endpoint as provided.
        missing: Unknown endpoint node IDs, in source-first order, without
            duplicates.
    """

    index: int
    source: str
    target: str
    missing: tuple[str, ...]


@dataclass(frozen=True)
class CycleGroupReport:
    """Result of :func:`cycle_group_report`.

    Attributes:
        groups: Cyclic strongly connected components, stably sorted by their
            sorted node IDs.
        invalid_edges: Edges referencing unknown nodes, sorted by input index.
            Their endpoints were never added to the analyzed graph.
    """

    groups: tuple[CyclicGroup, ...]
    invalid_edges: tuple[InvalidEdge, ...]


def _coerce_edge(edge: EdgeInput, index: int) -> tuple[str, str]:
    """Normalizes an input edge into a ``(source, target)`` tuple.

    Args:
        edge: The edge as a pair or a ``source``/``target`` mapping.
        index: Zero-based position of the edge in the input list, used in error
            messages.

    Returns:
        The normalized ``(source, target)`` pair.

    Raises:
        TypeError: If ``edge`` is neither a pair nor a mapping.
        ValueError: If ``edge`` is a malformed pair or mapping.
    """
    if isinstance(edge, Mapping):
        try:
            return edge["source"], edge["target"]
        except KeyError as exc:
            msg = f"Edge at position {index} is missing required key {exc.args[0]!r}"
            raise ValueError(msg) from exc

    if isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)):
        if len(edge) != EDGE_PAIR_LENGTH:
            msg = f"Edge at position {index} must be a (source, target) pair, got {len(edge)} elements"
            raise ValueError(msg)
        return edge[0], edge[1]

    msg = (
        f"Edge at position {index} must be a (source, target) pair or a "
        f"source/target mapping, got {type(edge).__name__}"
    )
    raise TypeError(msg)


def cycle_group_report(
    vertex_ids: Iterable[str],
    edges: Iterable[EdgeInput],
) -> CycleGroupReport:
    """Builds the cyclic strongly connected component report for a directed graph.

    Only the provided ``vertex_ids`` are analyzed: edges referencing a node not
    in ``vertex_ids`` are collected in ``invalid_edges`` (with their input
    position) and never cause the unknown node to join the graph.

    Args:
        vertex_ids: The known node IDs. Duplicate IDs are treated as one node.
        edges: Directed edges as ``(source, target)`` pairs or
            ``{"source": ..., "target": ...}`` mappings, in input order.

    Returns:
        A :class:`CycleGroupReport` with every cyclic SCC and its internal
        edges, plus the invalid edges with their positions. An acyclic graph
        yields an empty ``groups`` tuple.

    Raises:
        TypeError: If an edge is neither a pair nor a mapping.
        ValueError: If an edge is malformed (wrong pair length or missing a
            ``source``/``target`` key).
    """
    known_nodes = set(vertex_ids)

    invalid_edges: list[InvalidEdge] = []
    valid_edges: list[tuple[str, str]] = []
    for index, raw_edge in enumerate(edges):
        source, target = _coerce_edge(raw_edge, index)

        # Source-first order, deduplicated so an unknown self-loop is reported
        # once instead of twice.
        missing: list[str] = []
        if source not in known_nodes:
            missing.append(source)
        if target not in known_nodes and target != source:
            missing.append(target)

        if missing:
            invalid_edges.append(InvalidEdge(index=index, source=source, target=target, missing=tuple(missing)))
            continue

        valid_edges.append((source, target))

    # Construct the graph explicitly from the known nodes so rejected edges can
    # never smuggle unknown endpoints into the SCC analysis.
    graph = nx.DiGraph()
    graph.add_nodes_from(known_nodes)
    graph.add_edges_from(valid_edges)

    groups: list[CyclicGroup] = []
    for component in nx.strongly_connected_components(graph):
        nodes = sorted(component)
        has_cycle = len(nodes) > 1 or graph.has_edge(nodes[0], nodes[0])
        if not has_cycle:
            continue
        # A component subgraph contains exactly the edges internal to the SCC,
        # self-loops included. DiGraph collapses duplicate pairs, so the report
        # lists distinct edges.
        internal_edges = tuple(sorted(graph.subgraph(component).edges()))
        groups.append(CyclicGroup(nodes=tuple(nodes), edges=internal_edges))

    # SCC iteration order is not guaranteed; sorting by the sorted node tuple
    # yields a deterministic, ID-based group order.
    groups.sort(key=lambda group: group.nodes)
    invalid_edges.sort(key=lambda invalid: invalid.index)

    return CycleGroupReport(groups=tuple(groups), invalid_edges=tuple(invalid_edges))

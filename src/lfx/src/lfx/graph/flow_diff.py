"""Structural change report between two parsed flow payloads.

This module is intentionally side-effect free: it never builds a ``Graph``,
instantiates components, or mutates its inputs. The report only carries
node/edge IDs and edge endpoint field names — node parameters, custom code,
and credentials are never copied into the report.

The expected payload shape is the parsed ``GraphData`` mapping::

    {"nodes": [{"id": ..., ...}], "edges": [{"id": ..., "source": ..., ...}]}

Payloads wrapped as ``{"data": GraphData}`` are accepted as well. Missing
``nodes``/``edges`` keys are treated as empty lists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from typing_extensions import TypedDict

#: Edge identity fields inspected on retained edges, in report order.
EDGE_ENDPOINT_FIELDS: tuple[str, ...] = ("source", "target", "sourceHandle", "targetHandle")

#: Labels for the two compared flows, used in error messages.
_FLOW_LABELS: tuple[str, str] = ("old", "new")

FlowPayload: TypeAlias = Mapping[str, Any]
"""A parsed flow payload (``GraphData``) or a ``{"data": GraphData}`` wrapper."""


class NodeChanges(TypedDict):
    """Node IDs added to, removed from, or retained in the new flow."""

    added: list[str]
    removed: list[str]
    retained: list[str]


class RetainedEdgeChange(TypedDict):
    """A retained edge and the endpoint fields whose identifiers changed."""

    id: str
    changed_fields: list[str]


class EdgeChanges(TypedDict):
    """Edge IDs added/removed and per-edge endpoint changes for retained edges."""

    added: list[str]
    removed: list[str]
    retained: list[RetainedEdgeChange]


class FlowStructureDiff(TypedDict):
    """Structural change report for nodes and edges."""

    nodes: NodeChanges
    edges: EdgeChanges


class DuplicateFlowElementIdError(ValueError):
    """Raised when an input flow declares the same node/edge id more than once.

    Comparison stops before any result is produced.

    Attributes:
        flow: Which input flow contained the duplicate (``"old"`` or ``"new"``).
        kind: Element kind (``"node"`` or ``"edge"``).
        duplicate_id: The duplicated id.
        positions: Zero-based positions of every occurrence in the flow's
            ``nodes`` or ``edges`` list.
    """

    def __init__(
        self,
        *,
        flow: str,
        kind: str,
        duplicate_id: Any,
        positions: Sequence[int],
    ) -> None:
        self.flow = flow
        self.kind = kind
        self.duplicate_id = duplicate_id
        self.positions = list(positions)
        super().__init__(
            f"Duplicate {kind} id {duplicate_id!r} in the {flow} flow at positions {self.positions}; "
            "structural comparison stopped"
        )


def _unwrap_flow(flow: FlowPayload) -> FlowPayload:
    """Return the ``GraphData`` mapping, unwrapping a ``{"data": ...}`` payload."""
    if "nodes" in flow or "edges" in flow:
        return flow
    wrapped = flow.get("data")
    if isinstance(wrapped, Mapping) and ("nodes" in wrapped or "edges" in wrapped):
        return wrapped
    return flow


def _extract_elements(flow: FlowPayload, key: str, flow_label: str) -> Sequence[Any]:
    """Return the ``nodes``/``edges`` list of a flow, defaulting to an empty list."""
    elements = flow.get(key)
    if elements is None:
        return ()
    if not isinstance(elements, Sequence) or isinstance(elements, str | bytes):
        msg = f"The {flow_label} flow {key} list must be a sequence (found {type(elements).__name__})"
        raise TypeError(msg)
    return elements


def _index_elements(
    flow: FlowPayload,
    key: str,
    kind: str,
    flow_label: str,
) -> dict[Any, Mapping[str, Any]]:
    """Index nodes/edges by id, raising on duplicate or malformed elements.

    Args:
        flow: The parsed (and unwrapped) flow payload.
        key: ``"nodes"`` or ``"edges"``.
        kind: ``"node"`` or ``"edge"`` for error messages.
        flow_label: ``"old"`` or ``"new"`` for error messages.

    Returns:
        A mapping from element id to the original element mapping.

    Raises:
        DuplicateFlowElementIdError: If the same id occurs more than once.
        TypeError: If an element is not a mapping.
    """
    positions_by_id: dict[Any, list[int]] = {}
    elements = _extract_elements(flow, key, flow_label)
    for position, element in enumerate(elements):
        if not isinstance(element, Mapping):
            msg = f"{kind.capitalize()} at position {position} in the {flow_label} flow must be an object"
            raise TypeError(msg)
        positions_by_id.setdefault(element.get("id"), []).append(position)

    # Dict iteration follows first-occurrence order, so the first duplicated
    # id is reported deterministically.
    for duplicate_id, positions in positions_by_id.items():
        if len(positions) > 1:
            raise DuplicateFlowElementIdError(
                flow=flow_label,
                kind=kind,
                duplicate_id=duplicate_id,
                positions=positions,
            )

    return {element_id: elements[positions[0]] for element_id, positions in positions_by_id.items()}


def _edge_endpoint(edge: Mapping[str, Any], field: str) -> Any:
    """Read an edge endpoint field.

    Top-level ``source``/``target``/``sourceHandle``/``targetHandle`` keys are
    authoritative; payloads that only nest handles under ``data`` fall back to
    ``edge["data"][field]``. Values are returned as-is (handle identifiers may
    be strings or objects) and never normalized, keeping comparison pure.
    """
    if field in edge:
        return edge[field]
    nested = edge.get("data")
    if isinstance(nested, Mapping):
        return nested.get(field)
    return None


def _diff_id_sets(
    old_index: Mapping[Any, Any],
    new_index: Mapping[Any, Any],
) -> NodeChanges:
    """Build the added/removed/retained ID lists for one element kind."""
    old_ids = set(old_index)
    new_ids = set(new_index)
    return {
        "added": sorted(new_ids - old_ids, key=str),
        "removed": sorted(old_ids - new_ids, key=str),
        "retained": sorted(old_ids & new_ids, key=str),
    }


def diff_flow_structure(old_flow: FlowPayload, new_flow: FlowPayload) -> FlowStructureDiff:
    """Report structural additions, removals, and retained items between flows.

    Nodes and edges are compared by ID independently. For each retained edge,
    ``source``, ``target``, ``sourceHandle`` and ``targetHandle`` identifiers
    are compared and the names of the changed fields are listed (the values
    themselves are not reported). No node parameters, code, or credentials are
    included in the result.

    Inputs are never mutated, no ``Graph`` is built, and no component is
    instantiated. Duplicate node/edge IDs in either input stop the comparison:
    the first duplicate is reported with all of its list positions.

    Args:
        old_flow: The earlier parsed flow payload.
        new_flow: The later parsed flow payload.

    Returns:
        A report whose node/edge lists are stably sorted by ID.

    Raises:
        TypeError: If either input is not a parsed mapping or contains
            malformed node/edge entries.
        DuplicateFlowElementIdError: If a node or edge id is duplicated in
            either flow; the report is not produced in that case.
    """
    flows: dict[str, FlowPayload] = {}
    for flow_label, flow in zip(_FLOW_LABELS, (old_flow, new_flow), strict=True):
        if not isinstance(flow, Mapping):
            msg = f"The {flow_label} flow must be a parsed mapping (found {type(flow).__name__})"
            raise TypeError(msg)
        flows[flow_label] = _unwrap_flow(flow)

    # Validate old nodes/edges before new ones; within a flow, nodes first.
    old_nodes = _index_elements(flows["old"], "nodes", "node", "old")
    old_edges = _index_elements(flows["old"], "edges", "edge", "old")
    new_nodes = _index_elements(flows["new"], "nodes", "node", "new")
    new_edges = _index_elements(flows["new"], "edges", "edge", "new")

    retained_edges: list[RetainedEdgeChange] = []
    for edge_id in sorted(set(old_edges) & set(new_edges), key=str):
        old_edge = old_edges[edge_id]
        new_edge = new_edges[edge_id]
        changed_fields = [
            field
            for field in EDGE_ENDPOINT_FIELDS
            if _edge_endpoint(old_edge, field) != _edge_endpoint(new_edge, field)
        ]
        retained_edges.append({"id": edge_id, "changed_fields": changed_fields})

    return {
        "nodes": _diff_id_sets(old_nodes, new_nodes),
        "edges": {
            "added": sorted(set(new_edges) - set(old_edges), key=str),
            "removed": sorted(set(old_edges) - set(new_edges), key=str),
            "retained": retained_edges,
        },
    }

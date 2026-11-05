import copy

import pytest
from lfx.graph.flow_diff import (
    EDGE_ENDPOINT_FIELDS,
    DuplicateFlowElementIdError,
    diff_flow_structure,
)

# Fake values used to assert sensitive node payload never appears in reports.
FAKE_API_KEY = "sk-super-secret"
FAKE_CODE = "print('secret code')"


def node(node_id, **extra):
    """Build a node carrying fake sensitive payload that must never be reported."""
    return {
        "id": node_id,
        "type": "genericNode",
        "position": {"x": 0, "y": 0},
        "data": {
            "type": "OpenAIModel",
            "node": {
                "template": {
                    "api_key": {"type": "credential", "value": FAKE_API_KEY},
                    "code": {"type": "code", "value": FAKE_CODE},
                    "temperature": {"type": "float", "value": 0.1},
                }
            },
        },
        **extra,
    }


def edge(edge_id, source="a", target="b", source_handle="sh-a", target_handle="th-b"):
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
    }


def flow(nodes=None, edges=None):
    return {"nodes": nodes or [], "edges": edges or []}


# --------------------------------------------------------------------------- #
# Empty flows
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "empty_flow",
    [
        {},
        {"nodes": [], "edges": []},
        {"nodes": None, "edges": None},
        {"viewport": {"x": 0, "y": 0, "zoom": 1}},
    ],
)
def test_two_empty_flows_report_no_changes(empty_flow):
    report = diff_flow_structure(copy.deepcopy(empty_flow), copy.deepcopy(empty_flow))

    assert report == {
        "nodes": {"added": [], "removed": [], "retained": []},
        "edges": {"added": [], "removed": [], "retained": []},
    }


def test_empty_old_flow_reports_everything_as_added():
    new_flow = flow(nodes=[node("n1")], edges=[edge("e1")])

    report = diff_flow_structure({}, new_flow)

    assert report["nodes"] == {"added": ["n1"], "removed": [], "retained": []}
    assert report["edges"]["added"] == ["e1"]
    assert report["edges"]["removed"] == []
    assert report["edges"]["retained"] == []


def test_empty_new_flow_reports_everything_as_removed():
    old_flow = flow(nodes=[node("n1")], edges=[edge("e1")])

    report = diff_flow_structure(old_flow, {})

    assert report["nodes"] == {"added": [], "removed": ["n1"], "retained": []}
    assert report["edges"]["added"] == []
    assert report["edges"]["removed"] == ["e1"]
    assert report["edges"]["retained"] == []


def test_wrapped_data_payload_is_accepted():
    old_flow = {"data": flow(nodes=[node("n1")], edges=[edge("e1")])}
    new_flow = {"data": flow(nodes=[node("n1"), node("n2")], edges=[edge("e1"), edge("e2")])}

    report = diff_flow_structure(old_flow, new_flow)

    assert report["nodes"]["added"] == ["n2"]
    assert report["nodes"]["retained"] == ["n1"]
    assert report["edges"]["added"] == ["e2"]
    assert report["edges"]["retained"] == [{"id": "e1", "changed_fields": []}]


# --------------------------------------------------------------------------- #
# Nodes: add / remove / retain / reorder
# --------------------------------------------------------------------------- #


def test_node_added_removed_and_retained_are_reported():
    old_flow = flow(nodes=[node("keep-1"), node("gone"), node("keep-2")])
    new_flow = flow(nodes=[node("keep-2"), node("keep-1"), node("born")])

    report = diff_flow_structure(old_flow, new_flow)

    assert report["nodes"] == {
        "added": ["born"],
        "removed": ["gone"],
        "retained": ["keep-1", "keep-2"],
    }


def test_node_rearrangement_reports_all_retained_sorted_by_id():
    ids = [f"n{i}" for i in range(5)]
    old_flow = flow(nodes=[node(i) for i in ids])
    new_flow = flow(nodes=[node(i) for i in reversed(ids)])

    report = diff_flow_structure(old_flow, new_flow)

    assert report["nodes"]["added"] == []
    assert report["nodes"]["removed"] == []
    assert report["nodes"]["retained"] == sorted(ids)


def test_results_are_stably_sorted_by_id_even_with_scrambled_input():
    old_flow = flow(nodes=[node(i) for i in ("z", "a", "m", "b")])
    new_flow = flow(nodes=[node(i) for i in ("y", "m", "a", "c")])

    report = diff_flow_structure(old_flow, new_flow)

    assert report["nodes"]["added"] == ["c", "y"]
    assert report["nodes"]["removed"] == ["b", "z"]
    assert report["nodes"]["retained"] == ["a", "m"]


# --------------------------------------------------------------------------- #
# Edges: add / remove / retain and endpoint changes
# --------------------------------------------------------------------------- #


def test_edge_added_removed_and_unchanged_retained():
    old_flow = flow(edges=[edge("keep"), edge("gone")])
    new_flow = flow(edges=[edge("keep"), edge("born")])

    report = diff_flow_structure(old_flow, new_flow)

    assert report["edges"]["added"] == ["born"]
    assert report["edges"]["removed"] == ["gone"]
    assert report["edges"]["retained"] == [{"id": "keep", "changed_fields": []}]


def test_retained_edge_reconnected_to_other_source_and_target():
    old_flow = flow(edges=[edge("e1", source="a", target="b")])
    new_flow = flow(edges=[edge("e1", source="c", target="d")])

    report = diff_flow_structure(old_flow, new_flow)

    assert report["edges"]["retained"] == [{"id": "e1", "changed_fields": ["source", "target"]}]


def test_retained_edge_handle_changes_list_handle_field_names_only():
    old_flow = flow(
        edges=[
            edge("source-handle", source_handle="out-a"),
            edge("target-handle", target_handle="in-b"),
            edge("both-handles", source_handle="out-a", target_handle="in-b"),
        ]
    )
    new_flow = flow(
        edges=[
            edge("source-handle", source_handle="out-z"),
            edge("target-handle", target_handle="in-z"),
            edge("both-handles", source_handle="out-z", target_handle="in-z"),
        ]
    )

    report = diff_flow_structure(old_flow, new_flow)

    retained = {item["id"]: item["changed_fields"] for item in report["edges"]["retained"]}
    assert retained == {
        "source-handle": ["sourceHandle"],
        "target-handle": ["targetHandle"],
        "both-handles": ["sourceHandle", "targetHandle"],
    }


def test_retained_edges_are_sorted_by_id():
    old_flow = flow(edges=[edge(i) for i in ("e3", "e1", "e2")])
    new_flow = flow(edges=[edge(i) for i in ("e2", "e3", "e1")])

    report = diff_flow_structure(old_flow, new_flow)

    assert [item["id"] for item in report["edges"]["retained"]] == ["e1", "e2", "e3"]


def test_changed_fields_follow_canonical_order_regardless_of_payload_order():
    old_edge = {"id": "e1", "target": "b", "targetHandle": "th-b", "source": "a", "sourceHandle": "sh-a"}
    new_edge = {"id": "e1", "sourceHandle": "sh-z", "source": "z", "targetHandle": "th-z", "target": "z"}
    report = diff_flow_structure(flow(edges=[old_edge]), flow(edges=[new_edge]))

    assert report["edges"]["retained"][0]["changed_fields"] == list(EDGE_ENDPOINT_FIELDS)


def test_dict_handle_identifiers_are_compared_structurally():
    old_flow = flow(
        edges=[
            {
                "id": "e1",
                "source": "a",
                "target": "b",
                "sourceHandle": {"dataType": "Prompt", "id": "a", "name": "prompt"},
                "targetHandle": {"fieldName": "in", "id": "b"},
            }
        ]
    )
    new_flow = flow(
        edges=[
            {
                "id": "e1",
                "source": "a",
                "target": "b",
                "sourceHandle": {"dataType": "Prompt", "id": "a", "name": "prompt"},
                "targetHandle": {"fieldName": "other_in", "id": "b"},
            }
        ]
    )

    report = diff_flow_structure(old_flow, new_flow)

    assert report["edges"]["retained"] == [{"id": "e1", "changed_fields": ["targetHandle"]}]


def test_handles_nested_under_data_are_used_when_top_level_is_absent():
    old_flow = flow(
        edges=[
            {
                "id": "e1",
                "source": "a",
                "target": "b",
                "data": {"sourceHandle": "out-a", "targetHandle": "in-b"},
            }
        ]
    )
    new_flow = flow(
        edges=[
            {
                "id": "e1",
                "source": "a",
                "target": "b",
                "data": {"sourceHandle": "out-a", "targetHandle": "in-z"},
            }
        ]
    )

    report = diff_flow_structure(old_flow, new_flow)

    assert report["edges"]["retained"] == [{"id": "e1", "changed_fields": ["targetHandle"]}]


# --------------------------------------------------------------------------- #
# Privacy: node parameters, code and credentials never leak
# --------------------------------------------------------------------------- #


def test_report_never_contains_node_parameters_code_or_credentials():
    old_flow = flow(nodes=[node("n1")], edges=[edge("e1")])
    new_flow = flow(nodes=[node("n1", position={"x": 100, "y": 200})], edges=[edge("e1")])

    serialized = repr(diff_flow_structure(old_flow, new_flow))

    assert FAKE_API_KEY not in serialized
    assert FAKE_CODE not in serialized
    assert "temperature" not in serialized


def test_edge_endpoint_values_are_not_reported_only_field_names():
    old_flow = flow(edges=[edge("e1", source="secret-source-node")])
    new_flow = flow(edges=[edge("e1", source="other-secret-node")])

    report = diff_flow_structure(old_flow, new_flow)

    serialized = repr(report["edges"]["retained"])
    assert "secret-source-node" not in serialized
    assert "other-secret-node" not in serialized
    assert report["edges"]["retained"][0]["changed_fields"] == ["source"]


# --------------------------------------------------------------------------- #
# Duplicate IDs: report positions and stop
# --------------------------------------------------------------------------- #


def test_duplicate_node_id_in_old_flow_reports_positions_and_stops():
    old_flow = flow(nodes=[node("dup", position={"x": 0}), node("other"), node("dup", position={"x": 2})])
    new_flow = flow(nodes=[node("dup"), node("other"), node("extra")])

    with pytest.raises(DuplicateFlowElementIdError) as exc_info:
        diff_flow_structure(old_flow, new_flow)

    error = exc_info.value
    assert error.flow == "old"
    assert error.kind == "node"
    assert error.duplicate_id == "dup"
    assert error.positions == [0, 2]
    message = str(error)
    assert "0" in message
    assert "2" in message


def test_duplicate_node_id_in_new_flow_reports_positions_and_stops():
    old_flow = flow(nodes=[node("dup")])
    new_flow = flow(nodes=[node("dup"), node("x"), node("dup"), node("dup")])

    with pytest.raises(DuplicateFlowElementIdError) as exc_info:
        diff_flow_structure(old_flow, new_flow)

    error = exc_info.value
    assert error.flow == "new"
    assert error.kind == "node"
    assert error.duplicate_id == "dup"
    assert error.positions == [0, 2, 3]


def test_duplicate_edge_id_reports_edge_kind_and_positions():
    old_flow = flow(edges=[edge("e1"), edge("e2"), edge("e1")])

    with pytest.raises(DuplicateFlowElementIdError) as exc_info:
        diff_flow_structure(old_flow, flow())

    error = exc_info.value
    assert error.flow == "old"
    assert error.kind == "edge"
    assert error.duplicate_id == "e1"
    assert error.positions == [0, 2]


def test_first_duplicate_is_reported_in_first_occurrence_order():
    bad_flow = flow(nodes=[node("a"), node("b"), node("a"), node("b")])

    with pytest.raises(DuplicateFlowElementIdError) as exc_info:
        diff_flow_structure(bad_flow, flow())

    assert exc_info.value.duplicate_id == "a"
    assert exc_info.value.positions == [0, 2]


def test_old_flow_duplicates_take_precedence_over_new_flow():
    old_flow = flow(nodes=[node("dup"), node("dup")])
    new_flow = flow(nodes=[node("dup"), node("dup")])

    with pytest.raises(DuplicateFlowElementIdError) as exc_info:
        diff_flow_structure(old_flow, new_flow)

    assert exc_info.value.flow == "old"


def test_nodes_are_validated_before_edges():
    bad_flow = flow(
        nodes=[node("dup"), node("dup")],
        edges=[edge("dup-edge"), edge("dup-edge")],
    )

    with pytest.raises(DuplicateFlowElementIdError) as exc_info:
        diff_flow_structure(bad_flow, flow())

    assert exc_info.value.kind == "node"


def test_duplicate_ids_are_value_error_subclasses():
    with pytest.raises(ValueError, match="Duplicate node id"):
        diff_flow_structure(flow(nodes=[node("x"), node("x")]), flow())


# --------------------------------------------------------------------------- #
# Inputs are not mutated and malformed input is rejected
# --------------------------------------------------------------------------- #


def test_inputs_are_not_mutated():
    old_flow = flow(nodes=[node("n1")], edges=[edge("e1", source="a", target="b")])
    new_flow = flow(nodes=[node("n1"), node("n2")], edges=[edge("e1", source="a", target="c")])
    old_snapshot = copy.deepcopy(old_flow)
    new_snapshot = copy.deepcopy(new_flow)

    diff_flow_structure(old_flow, new_flow)

    assert old_flow == old_snapshot
    assert new_flow == new_snapshot


def test_non_mapping_flow_raises_type_error():
    with pytest.raises(TypeError, match="old flow"):
        diff_flow_structure([], flow())
    with pytest.raises(TypeError, match="new flow"):
        diff_flow_structure(flow(), None)


def test_non_sequence_node_list_raises_type_error():
    with pytest.raises(TypeError, match="nodes"):
        diff_flow_structure({"nodes": {"n1": {}}}, flow())


def test_non_object_node_raises_type_error():
    with pytest.raises(TypeError, match="position 1"):
        diff_flow_structure(flow(nodes=[node("n1"), "not-a-node"]), flow())

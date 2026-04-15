from __future__ import annotations

from simlab.autoresearch.analysis import build_analysis
from simlab.autoresearch.analysis import compact_tool_error_taxonomy


def test_build_analysis_compacts_tool_error_taxonomy_dict() -> None:
    analysis = build_analysis(
        objective_type="pass_rate",
        objective_target=0.8,
        best_iteration=0,
        best_result={"objective_value": 0.0},
        best_eval={
            "summary": {
                "tool_error_taxonomy": {
                    "total_errors": 12,
                    "categories": [{"category": "timeout", "count": 2}] * 20,
                    "most_error_prone_tools": [{"tool": "calendar", "error_count": 1}] * 20,
                    "top_error_messages": [{"message": "boom", "count": 1}] * 20,
                }
            }
        },
        latest_iteration=0,
        latest_result={"objective_value": 0.0},
        latest_eval={"summary": {}},
        history=[],
    )

    taxonomy = analysis.get("tool_error_taxonomy")
    assert isinstance(taxonomy, dict)
    assert taxonomy.get("total_errors") == 12
    assert isinstance(taxonomy.get("categories"), list)
    assert len(taxonomy.get("categories") or []) == 10
    assert isinstance(taxonomy.get("most_error_prone_tools"), list)
    assert len(taxonomy.get("most_error_prone_tools") or []) == 10
    assert isinstance(taxonomy.get("top_error_messages"), list)
    assert len(taxonomy.get("top_error_messages") or []) == 10


def test_compact_tool_error_taxonomy_rejects_bool_total_errors() -> None:
    compacted = compact_tool_error_taxonomy(
        {
            "total_errors": True,
            "categories": [],
            "most_error_prone_tools": [],
            "top_error_messages": [],
        },
        10,
    )
    assert compacted["total_errors"] == 0

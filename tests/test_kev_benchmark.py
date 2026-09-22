import json
from pathlib import Path

import pytest

from tools import kev_benchmark

CONFIG = Path(__file__).parents[1] / "configs/kev_real_weight_acceptance.json"


def test_acceptance_config_has_required_sweeps_and_thresholds():
    config = kev_benchmark.read_json(CONFIG)
    plan = kev_benchmark.build_plan(config)

    assert len(plan) == 10
    assert plan[0]["name"] == "s128-q1-k2-c1"
    assert any(case["family"] == "multi" for case in plan)


def test_modified_acceptance_threshold_is_rejected():
    config = kev_benchmark.read_json(CONFIG)
    config["thresholds"]["max_probability_absolute_error"] = 0.2

    with pytest.raises(ValueError, match="must remain 0.02"):
        kev_benchmark.validate_config(config)


def test_load_verified_states_rejects_duplicate_counts(tmp_path):
    path = tmp_path / "states.json"
    path.write_text(
        json.dumps(
            [
                {"state_tokens": 128, "state": "one"},
                {"state_tokens": 128, "state": "two"},
            ]
        )
    )

    with pytest.raises(ValueError, match="unique positive"):
        kev_benchmark.load_verified_states(path)


def test_make_request_preserves_requested_shape():
    request = kev_benchmark.make_request("kev", {"中文": True}, 4, 8)

    assert request["state"] == {"中文": True}
    assert len(request["questions"]) == 4
    assert all(
        len(question["criteria"]) == 8 for question in request["questions"].values()
    )


def test_parse_prometheus_actual_token_counter():
    metrics = """
# HELP vllm:systemone_input_tokens_total test
vllm:systemone_input_tokens_total{kind="logical",model="kev"} 12
vllm:systemone_input_tokens_total{model="kev",kind="actual"} 34
vllm:systemone_input_tokens_total{model="other",kind="actual"} 56
"""

    assert (
        kev_benchmark.parse_prometheus_metric(
            metrics,
            "vllm:systemone_input_tokens_total",
            {"model": "kev", "kind": "actual"},
        )
        == 34
    )


def test_run_case_reports_success_and_keeps_external_metrics_unclaimed():
    calls = 0

    def post(_url, request, _timeout):
        nonlocal calls
        calls += 1
        return {
            "answers": {name: {} for name in request["questions"]},
            "usage": {"input_tokens": 11},
        }

    request = kev_benchmark.make_request("kev", "state", 2, 2)
    report = kev_benchmark.run_case(
        endpoint="http://unused",
        request=request,
        question_count=2,
        concurrency=2,
        warmup_requests=1,
        measured_requests=4,
        timeout=1,
        post=post,
    )

    assert calls == 5
    assert report["successful_requests"] == 4
    assert report["failed_requests"] == 0
    assert report["requests_per_second"] > 0
    assert report["questions_per_second"] > 0
    assert report["actual_input_tokens_per_second"] is None
    assert report["peak_device_memory_mib"] is None


def test_run_case_uses_actual_token_counter_delta():
    counters = iter([100.0, 140.0])
    request = kev_benchmark.make_request("kev", "state", 1, 2)
    report = kev_benchmark.run_case(
        endpoint="http://unused",
        request=request,
        question_count=1,
        concurrency=1,
        warmup_requests=0,
        measured_requests=1,
        timeout=1,
        post=lambda *_args: {
            "answers": {"question-0": {}},
            "usage": {"input_tokens": 10},
        },
        actual_token_counter=lambda: next(counters),
    )

    assert report["actual_input_tokens_per_second"] > 0
    assert report["missing_external_measurements"] == [
        "peak device memory from the Ascend runtime"
    ]


def test_run_case_records_atomic_response_failure():
    request = kev_benchmark.make_request("kev", "state", 2, 2)
    report = kev_benchmark.run_case(
        endpoint="http://unused",
        request=request,
        question_count=2,
        concurrency=1,
        warmup_requests=0,
        measured_requests=1,
        timeout=1,
        post=lambda *_args: {"answers": {"question-0": {}}},
    )

    assert report["successful_requests"] == 0
    assert report["failed_requests"] == 1
    assert "incomplete answer set" in report["failures"][0]

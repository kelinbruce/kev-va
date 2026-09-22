"""Plan and run reproducible SystemOne HTTP benchmark cases.

The tool never downloads a tokenizer or model. State payloads and their verified
token counts must be supplied by the acceptance environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import statistics
import time
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REQUIRED_SWEEPS = {
    "state_tokens": [128, 1024, 4096],
    "question_count": [1, 4, 16],
    "option_count": [2, 8, 32],
    "concurrency": [1, 4, 16],
}
PROMETHEUS_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("format_version") != 1:
        raise ValueError("unsupported acceptance config format")
    model = config.get("model", {})
    for name in ("source_revision", "base_revision", "reference_revision"):
        value = model.get(name)
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"model.{name} must be a full commit SHA")
    benchmark = config.get("benchmark", {})
    baseline = benchmark.get("baseline", {})
    for dimension, required in REQUIRED_SWEEPS.items():
        actual = benchmark.get("single_factor", {}).get(dimension)
        if actual != required:
            raise ValueError(f"benchmark sweep {dimension} must be {required}")
        if baseline.get(dimension) not in required:
            raise ValueError(f"benchmark baseline {dimension} is outside sweep")
    if not benchmark.get("multi_factor"):
        raise ValueError("benchmark must contain a multi-factor case")
    thresholds = config.get("thresholds", {})
    expected = {
        "mean_probability_absolute_error": 0.002,
        "max_probability_absolute_error": 0.02,
        "argmax_agreement": 0.99,
        "batch_max_probability_delta": 0.005,
    }
    for name, limit in expected.items():
        if thresholds.get(name) != limit:
            raise ValueError(f"threshold {name} must remain {limit}")


def build_plan(config: dict[str, Any]) -> list[dict[str, int | str]]:
    validate_config(config)
    benchmark = config["benchmark"]
    baseline = benchmark["baseline"]
    cases: list[dict[str, int | str]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add(values: dict[str, int], family: str) -> None:
        key = tuple(values[name] for name in REQUIRED_SWEEPS)
        if key in seen:
            return
        seen.add(key)
        case = dict(values)
        case["family"] = family
        case["name"] = "s{}-q{}-k{}-c{}".format(*key)
        cases.append(case)

    add(dict(baseline), "baseline")
    for dimension, values in benchmark["single_factor"].items():
        for value in values:
            case = dict(baseline)
            case[dimension] = value
            add(case, f"single:{dimension}")
    for case in benchmark["multi_factor"]:
        add(dict(case), "multi")
    return cases


def load_verified_states(path: Path) -> dict[int, Any]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise TypeError("verified state workload must be a JSON list")
    states: dict[int, Any] = {}
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"state_tokens", "state"}:
            raise ValueError("each workload entry needs state_tokens and state")
        count = item["state_tokens"]
        if type(count) is not int or count <= 0 or count in states:
            raise ValueError("state_tokens must be unique positive integers")
        states[count] = item["state"]
    return states


def make_request(
    model: str, state: Any, question_count: int, option_count: int
) -> dict[str, Any]:
    if question_count <= 0 or not 1 <= option_count <= 255:
        raise ValueError("invalid question or option count")
    questions = {}
    for question_index in range(question_count):
        criteria = {
            f"option-{option_index}": f"候选项 {option_index}"
            for option_index in range(option_count)
        }
        questions[f"question-{question_index}"] = {
            "type": "choice",
            "instructions": "选择最合适的候选项",
            "criteria": criteria,
        }
    return {"model": model, "state": state, "questions": questions}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate an empty percentile")
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def parse_prometheus_metric(
    text: str, name: str, required_labels: dict[str, str]
) -> float:
    values = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        series, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        metric_name, labels = series, {}
        if "{" in series and series.endswith("}"):
            metric_name, raw_labels = series.split("{", 1)
            labels = {
                key: bytes(value, "utf-8").decode("unicode_escape")
                for key, value in PROMETHEUS_LABEL.findall(raw_labels[:-1])
            }
        if metric_name == name and all(
            labels.get(key) == value for key, value in required_labels.items()
        ):
            values.append(float(raw_value))
    if not values:
        raise ValueError(f"metric not found: {name} {required_labels}")
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError(f"metric is not finite: {name}")
    return total


def read_actual_token_counter(url: str, model: str, timeout: float) -> float:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return parse_prometheus_metric(
        text,
        "vllm:systemone_input_tokens_total",
        {"model": model, "kind": "actual"},
    )


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def run_case(
    *,
    endpoint: str,
    request: dict[str, Any],
    question_count: int,
    concurrency: int,
    warmup_requests: int,
    measured_requests: int,
    timeout: float,
    post: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
    actual_token_counter: Callable[[], float] | None = None,
) -> dict[str, Any]:
    for _ in range(warmup_requests):
        response = post(endpoint, request, timeout)
        if len(response.get("answers", {})) != question_count:
            raise RuntimeError("warm-up returned an incomplete answer set")

    def invoke(_index: int) -> tuple[float, dict[str, Any]]:
        started = time.perf_counter()
        response = post(endpoint, request, timeout)
        return time.perf_counter() - started, response

    actual_tokens_before = actual_token_counter() if actual_token_counter else None
    started = time.perf_counter()
    successes: list[tuple[float, dict[str, Any]]] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke, index) for index in range(measured_requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                latency, response = future.result()
                if len(response.get("answers", {})) != question_count:
                    raise RuntimeError("incomplete answer set")
                successes.append((latency, response))
            # The benchmark must turn transport, HTTP and response validation
            # failures into one machine-readable result rather than aborting.
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started
    actual_tokens_after = actual_token_counter() if actual_token_counter else None
    latencies = [item[0] for item in successes]
    logical_tokens = sum(
        int(item[1].get("usage", {}).get("input_tokens", 0)) for item in successes
    )
    actual_tokens = None
    if actual_tokens_before is not None and actual_tokens_after is not None:
        actual_tokens = actual_tokens_after - actual_tokens_before
        if actual_tokens < 0 or not math.isfinite(actual_tokens):
            raise RuntimeError("actual token counter did not increase monotonically")
    missing = ["peak device memory from the Ascend runtime"]
    if actual_tokens is None:
        missing.insert(0, "actual token counter delta from /metrics")
    return {
        "measured_requests": measured_requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "latency_seconds": {
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
        },
        "requests_per_second": len(successes) / elapsed if elapsed else None,
        "questions_per_second": (
            len(successes) * question_count / elapsed if elapsed else None
        ),
        "logical_input_tokens_per_second": (
            logical_tokens / elapsed if elapsed else None
        ),
        "actual_input_tokens_per_second": (
            actual_tokens / elapsed if actual_tokens is not None and elapsed else None
        ),
        "peak_device_memory_mib": None,
        "missing_external_measurements": missing,
    }


def _select_case(
    cases: Iterable[dict[str, int | str]], name: str
) -> dict[str, int | str]:
    for case in cases:
        if case["name"] == name:
            return case
    raise ValueError(f"unknown benchmark case: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kev_real_weight_acceptance.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="print the required benchmark matrix")
    run_parser = subparsers.add_parser("run", help="run one benchmark case")
    run_parser.add_argument("--case", required=True)
    run_parser.add_argument("--states", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    plan = build_plan(config)
    if args.command == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    states_path = args.states or Path(config["datasets"]["state_workloads_json"])
    states = load_verified_states(states_path)
    case = _select_case(plan, args.case)
    state_tokens = int(case["state_tokens"])
    if state_tokens not in states:
        raise ValueError(f"no verified state payload for {state_tokens} tokens")
    request = make_request(
        config["model"]["model_id"],
        states[state_tokens],
        int(case["question_count"]),
        int(case["option_count"]),
    )
    benchmark = config["benchmark"]
    report = {
        "case": case,
        "model": config["model"],
        "result": run_case(
            endpoint=config["server"]["endpoint"],
            request=request,
            question_count=int(case["question_count"]),
            concurrency=int(case["concurrency"]),
            warmup_requests=int(benchmark["warmup_requests"]),
            measured_requests=int(benchmark["measured_requests"]),
            timeout=float(config["server"]["timeout_seconds"]),
            actual_token_counter=lambda: read_actual_token_counter(
                config["server"]["metrics_endpoint"],
                config["model"]["model_id"],
                float(config["server"]["timeout_seconds"]),
            ),
        ),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Concurrent EnvBench-derived replay for a live vLLM-Continuum server.

The original EnvBench prompts cannot be reconstructed from the trajectory
events. This script uses deterministic token-id prompts that match each
event's input_tokens and preserve within-trajectory prefix relationships.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformers import AutoTokenizer


TTL_RE = re.compile(
    r"Continuum (?:dynamic|baseline) TTL request=(?P<request>\S+) "
    r"job=(?P<job>\S+) tool=(?P<tool>\S+) "
    r"ttl=(?P<ttl>[0-9.eE+-]+) source=(?P<source>\S+)"
)


def quantile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] * (1 - w) + ordered[hi] * w


def distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {
            "count": 0, "mean": None, "median": None,
            "p90": None, "p95": None, "p99": None, "max": None,
        }
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": quantile(clean, 0.90),
        "p95": quantile(clean, 0.95),
        "p99": quantile(clean, 0.99),
        "max": max(clean),
    }


def fetch_text(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def prom_values(text: str, name: str) -> list[float]:
    out: list[float] = []
    for line in text.splitlines():
        if not (line.startswith(name + " ") or line.startswith(name + "{")):
            continue
        try:
            out.append(float(line.rsplit(" ", 1)[1]))
        except ValueError:
            pass
    return out


def metric_sum(text: str, name: str) -> float:
    return sum(prom_values(text, name))


def metric_first(text: str, name: str) -> float | None:
    values = prom_values(text, name)
    return values[0] if values else None


def delta(after: float, before: float) -> float:
    return max(0.0, after - before)


class MetricsPoller:
    def __init__(self, url: str, interval: float) -> None:
        self.url = url
        self.interval = interval
        self.rows: list[dict[str, float | None]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.start_perf = 0.0

    def start(self) -> None:
        self.start_perf = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def sample(self) -> None:
        try:
            text = fetch_text(self.url)
            self.rows.append({
                "elapsed_seconds": time.perf_counter() - self.start_perf,
                "kv_cache_usage_perc": metric_first(
                    text, "vllm:kv_cache_usage_perc"
                ),
                "num_requests_running": metric_first(
                    text, "vllm:num_requests_running"
                ),
                "num_requests_waiting": metric_first(
                    text, "vllm:num_requests_waiting"
                ),
                # CONTINUUM_KV_POLLER_V1
                "continuum_pinned_requests": metric_first(
                    text, "vllm:continuum_pinned_requests"
                ),
                "continuum_pinned_blocks": metric_first(
                    text, "vllm:continuum_pinned_blocks"
                ),
                "continuum_running_blocks": metric_first(
                    text, "vllm:continuum_running_blocks"
                ),
                "continuum_shared_pinned_running_blocks": metric_first(
                    text, "vllm:continuum_shared_pinned_running_blocks"
                ),
                "continuum_active_blocks": metric_first(
                    text, "vllm:continuum_active_blocks"
                ),
                "continuum_active_unpinned_blocks": metric_first(
                    text, "vllm:continuum_active_unpinned_blocks"
                ),
                "continuum_free_blocks": metric_first(
                    text, "vllm:continuum_free_blocks"
                ),
                "continuum_true_free_blocks": metric_first(
                    text, "vllm:continuum_true_free_blocks"
                ),
                "continuum_evictable_cached_blocks": metric_first(
                    text, "vllm:continuum_evictable_cached_blocks"
                ),
                "continuum_total_blocks": metric_first(
                    text, "vllm:continuum_total_blocks"
                ),
                "continuum_pin_events_total": metric_first(
                    text, "vllm:continuum_pin_events_total"
                ),
                "continuum_unpin_events_total": metric_first(
                    text, "vllm:continuum_unpin_events_total"
                ),
                "continuum_evicted_blocks_total": metric_first(
                    text, "vllm:continuum_evicted_blocks_total"
                ),
            })
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(2.0, self.interval * 4))
        self.sample()


def load_rows(
    path: Path,
    max_events: int,
    min_tokens: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "global_event_index", "trajectory_event_index", "event_id",
            "trajectory", "environment", "agent_config", "tool_name",
            "duration_seconds", "input_tokens", "is_error",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

        for row in reader:
            if not row["input_tokens"].strip():
                continue
            input_tokens = int(float(row["input_tokens"]))
            duration = float(row["duration_seconds"])
            if not (min_tokens <= input_tokens <= max_tokens):
                continue
            if duration < 0 or not math.isfinite(duration):
                continue
            rows.append({
                "global_event_index": int(row["global_event_index"]),
                "trajectory_event_index": int(row["trajectory_event_index"]),
                "event_id": row["event_id"],
                "trajectory": row["trajectory"],
                "environment": row["environment"],
                "agent_config": row["agent_config"],
                "tool_name": row["tool_name"],
                "duration_seconds_raw": duration,
                "input_tokens_target": input_tokens,
                "is_error": int(row["is_error"]),
            })
            if len(rows) >= max_events:
                break
    rows.sort(key=lambda x: x["global_event_index"])
    if not rows:
        raise RuntimeError("No EnvBench rows matched the filters")
    return rows


def build_prompt_tokens(
    tokenizer: Any,
    trajectory_index: int,
    max_tokens: int,
    common_prefix_tokens: int,
) -> list[int]:
    common_text = (
        "You are an autonomous software engineering agent. "
        "Retain prior context and use tools carefully. "
    )
    unique_text = (
        f"Trajectory {trajectory_index}. Inspect the environment, reason "
        "about the state, execute the next action, and preserve observations. "
    )
    common = tokenizer.encode(
        common_text * 128, add_special_tokens=False
    )[: min(common_prefix_tokens, max_tokens)]
    unique = tokenizer.encode(unique_text * 256, add_special_tokens=False)
    if not unique:
        raise RuntimeError("Tokenizer produced no tokens")
    result = list(common)
    while len(result) < max_tokens:
        result.extend(unique[: max_tokens - len(result)])
    return result


def stream_request(
    url: str,
    model: str,
    prompt_ids: list[int],
    max_output_tokens: int,
    job_id: str,
    last_tool: str | None,
    current_tool: str | None,
    is_last_step: bool,
    request_id: str,
    tokenizer: Any,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "job_id": job_id,
        "last_func_call": last_tool,
        "this_func_call": current_tool,
        "is_last_step": is_last_step,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-Id": request_id,
        },
        method="POST",
    )

    start_wall = time.time()

    start = time.perf_counter()

    response_open: float | None = None

    first_raw_line: float | None = None

    first_sse_data: float | None = None

    first_choice: float | None = None

    first_token: float | None = None

    end = start



    def stage_fields(end_time: float) -> dict[str, float | None]:

        def elapsed(point: float | None) -> float | None:

            return None if point is None else point - start



        def wall_time(point: float | None) -> float | None:

            return (

                None

                if point is None

                else start_wall + point - start

            )



        return {

            "request_start_unix_seconds": start_wall,

            "response_open_unix_seconds": wall_time(response_open),

            "first_raw_line_unix_seconds": wall_time(first_raw_line),

            "first_sse_data_unix_seconds": wall_time(first_sse_data),

            "first_choice_unix_seconds": wall_time(first_choice),

            "first_token_unix_seconds": wall_time(first_token),

            "request_end_unix_seconds": wall_time(end_time),

            "response_open_seconds": elapsed(response_open),

            "first_raw_line_seconds": elapsed(first_raw_line),

            "first_sse_data_seconds": elapsed(first_sse_data),

            "first_choice_seconds": elapsed(first_choice),

            "first_token_seconds": elapsed(first_token),

            "open_to_first_token_seconds": (

                first_token - response_open

                if first_token is not None

                and response_open is not None

                else None

            ),

            "response_read_seconds": (

                end_time - response_open

                if response_open is not None

                else None

            ),

        }
    parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    status_code: int | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:

            response_open = time.perf_counter()

            status_code = resp.status
            for raw in resp:

                raw_time = time.perf_counter()

                if first_raw_line is None:

                    first_raw_line = raw_time

                line = raw.decode(

                    "utf-8", errors="replace"

                ).strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if not data:

                    continue

                if first_sse_data is None:

                    first_sse_data = time.perf_counter()

                chunk = json.loads(data)
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:

                    if first_choice is None:

                        first_choice = time.perf_counter()

                    choice = choices[0]
                    text = choice.get("text") or ""
                    if text:
                        if first_token is None:
                            first_token = time.perf_counter()
                        parts.append(text)
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
            end = time.perf_counter()
    except urllib.error.HTTPError as exc:
        end = time.perf_counter()
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "success": 0, "status_code": exc.code,
            "error": f"HTTPError {exc.code}: {body}",
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "ttft_seconds": None, "tpot_seconds": None,
            "e2e_seconds": end - start,

            "finish_reason": None,

            **stage_fields(end),

        }
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        return {
            "success": 0, "status_code": status_code,
            "error": f"{type(exc).__name__}: {exc}",
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "ttft_seconds": None, "tpot_seconds": None,
            "e2e_seconds": end - start,

            "finish_reason": None,

            **stage_fields(end),

        }
    text = "".join(parts)
    prompt_tokens = int(usage.get("prompt_tokens") or len(prompt_ids))
    completion_tokens = int(
        usage.get("completion_tokens")
        or len(tokenizer.encode(text, add_special_tokens=False))
    )
    total_tokens = int(
        usage.get("total_tokens") or prompt_tokens + completion_tokens
    )
    ttft = first_token - start if first_token is not None else None
    e2e = end - start
    tpot = None
    if first_token is not None and completion_tokens > 1:
        tpot = (end - first_token) / (completion_tokens - 1)

    return {
        "success": 1, "status_code": status_code, "error": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "e2e_seconds": e2e,
        "finish_reason": finish_reason,

        **stage_fields(end),

    }
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_ttl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = TTL_RE.search(line)
        if match:
            rows.append({
                "request_id": match.group("request"),
                "job_id": match.group("job"),
                "tool_name": match.group("tool"),
                "ttl_seconds": float(match.group("ttl")),
                "history_source": match.group("source"),
                "log_line": line,
            })
    return rows


def attach_ttl(
    events: list[dict[str, Any]],
    ttl_rows: list[dict[str, Any]],
) -> None:
    queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in ttl_rows:
        queues[row["job_id"]].append(row)

    for event in events:
        match = None
        queue = queues[event["job_id"]]
        while queue:
            candidate = queue.popleft()
            if candidate["tool_name"] == event["tool_name"]:
                match = candidate
                break
        if match is None:
            event.update({
                "ttl_seconds": None,
                "ttl_history_source": None,
                "ttl_hit": None,
                "ttl_timeout": None,
            })
            continue
        ttl = float(match["ttl_seconds"])
        duration = float(event["duration_seconds_effective"])
        event.update({
            "ttl_seconds": ttl,
            "ttl_history_source": match["history_source"],
            "ttl_hit": int(duration <= ttl),
            "ttl_timeout": int(duration > ttl),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/continuum/envbench_ttl_trace/envbench_ttl_events_all.csv",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-events", type=int, default=150)
    parser.add_argument("--min-input-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=3500)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--duration-scale", type=float, default=0.001)
    parser.add_argument("--max-sleep-seconds", type=float, default=0.2)
    parser.add_argument("--metrics-interval", type=float, default=0.2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--common-prefix-tokens", type=int, default=256)
    parser.add_argument(
        "--reset-aware-prefix",
        action="store_true",
        help=(
            "Start a new deterministic prefix stream whenever "
            "input_tokens decreases within a trajectory. "
            "Disabled by default for backward-compatible replay."
        ),
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Number of concurrent trajectory workers",
    )
    parser.add_argument(
        "--experiment-label", default="concurrent",
        help="Label included in the output directory name",
    )
    parser.add_argument(
        "--log-flush-wait", type=float, default=0.5,
        help="Seconds to wait for server.log to flush after the workload",
    )
    parser.add_argument(
        "--lane-assignment",
        choices=("round_robin", "balanced"),
        default="round_robin",
        help=(
            "Assign trajectories using original round-robin "
            "or load-aware balanced placement."
        ),
    )
    args = parser.parse_args()

    if args.max_events < 102:
        raise ValueError("--max-events must be at least 102")
    if args.duration_scale < 0 or args.max_sleep_seconds < 0:
        raise ValueError("Duration controls must be non-negative")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.log_flush_wait < 0:
        raise ValueError("--log-flush-wait must be non-negative")

    root = Path.cwd()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        latest = root / "results/continuum/runtime_full/latest_run_dir.txt"
        run_dir = Path(latest.read_text(encoding="utf-8").strip())

    server_log = run_dir / "server.log"
    if not server_log.exists():
        raise FileNotFoundError(f"Missing server log: {server_log}")

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.experiment_label)
    output_dir = run_dir / (
        f"envbench_runtime_{safe_label}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = load_rows(
        input_path, args.max_events,
        args.min_input_tokens, args.max_input_tokens,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    first_index: dict[str, int] = {}
    for row in rows:
        grouped[row["trajectory"]].append(row)
        first_index.setdefault(row["trajectory"], row["global_event_index"])
    trajectories = sorted(grouped, key=lambda x: first_index[x])

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    prompt_streams: dict[str, list[int]] = {}
    job_ids: dict[str, str] = {}
    for i, trajectory in enumerate(trajectories):
        max_target = max(
            row["input_tokens_target"] for row in grouped[trajectory]
        )
        prompt_streams[trajectory] = build_prompt_tokens(
            tokenizer, i, max_target, args.common_prefix_tokens
        )
        job_ids[trajectory] = f"envbench-runtime-{i:05d}"

    completions_url = args.base_url.rstrip("/") + "/v1/completions"
    metrics_url = args.base_url.rstrip("/") + "/metrics"
    metrics_before = fetch_text(metrics_url)
    (output_dir / "metrics_before.txt").write_text(
        metrics_before, encoding="utf-8"
    )
    log_offset = server_log.stat().st_size

    request_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    poller = MetricsPoller(metrics_url, args.metrics_interval)
    start_wall = time.time()
    start_perf = time.perf_counter()

    worker_count = min(args.concurrency, len(trajectories))

    lanes: list[list[tuple[int, str]]] = [

        [] for _ in range(worker_count)

    ]



    if args.lane_assignment == "round_robin":

        for trajectory_index, trajectory in enumerate(trajectories):

            lanes[trajectory_index % worker_count].append(

                (trajectory_index, trajectory)

            )

    else:

        trajectory_loads: dict[str, tuple[int, int, float]] = {}



        for trajectory in trajectories:

            items = grouped[trajectory]

            trajectory_loads[trajectory] = (

                len(items),

                sum(

                    item["input_tokens_target"]

                    for item in items

                ),

                sum(

                    min(

                        item["duration_seconds_raw"]

                        * args.duration_scale,

                        args.max_sleep_seconds,

                    )

                    for item in items

                ),

            )



        target_events = (

            sum(

                load[0]

                for load in trajectory_loads.values()

            )

            / worker_count

        )

        target_tokens = (

            sum(

                load[1]

                for load in trajectory_loads.values()

            )

            / worker_count

        )

        target_wait = (

            sum(

                load[2]

                for load in trajectory_loads.values()

            )

            / worker_count

        )



        def normalized(

            value: float,

            target: float,

        ) -> float:

            return value / target if target > 0 else 0.0



        ordered_trajectories = sorted(

            enumerate(trajectories),

            key=lambda pair: (

                max(

                    normalized(

                        trajectory_loads[pair[1]][0],

                        target_events,

                    ),

                    normalized(

                        trajectory_loads[pair[1]][1],

                        target_tokens,

                    ),

                ),

                normalized(

                    trajectory_loads[pair[1]][2],

                    target_wait,

                ),

                trajectory_loads[pair[1]][0],

                trajectory_loads[pair[1]][1],

            ),

            reverse=True,

        )



        lane_events = [0] * worker_count

        lane_tokens = [0] * worker_count

        lane_wait = [0.0] * worker_count



        for trajectory_index, trajectory in ordered_trajectories:

            event_count, token_count, effective_wait = (

                trajectory_loads[trajectory]

            )



            worker_index = min(

                range(worker_count),

                key=lambda index: (

                    max(

                        normalized(

                            lane_events[index] + event_count,

                            target_events,

                        ),

                        normalized(

                            lane_tokens[index] + token_count,

                            target_tokens,

                        ),

                    ),

                    normalized(

                        lane_events[index] + event_count,

                        target_events,

                    )

                    + normalized(

                        lane_tokens[index] + token_count,

                        target_tokens,

                    ),

                    normalized(

                        lane_wait[index] + effective_wait,

                        target_wait,

                    ),

                    index,

                ),

            )



            lanes[worker_index].append(

                (trajectory_index, trajectory)

            )

            lane_events[worker_index] += event_count

            lane_tokens[worker_index] += token_count

            lane_wait[worker_index] += effective_wait



        # Keep each lane's trajectories in original trace order.

        for lane in lanes:

            lane.sort(key=lambda item: item[0])



    start_event = threading.Event()
    abort_event = threading.Event()

    def run_lane(
        worker_index: int,
        lane: list[tuple[int, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        local_requests: list[dict[str, Any]] = []
        local_events: list[dict[str, Any]] = []
        start_event.wait()

        for trajectory_index, trajectory in lane:
            if abort_event.is_set():
                return local_requests, local_events

            items = sorted(
                grouped[trajectory],
                key=lambda x: x["trajectory_event_index"],
            )
            job_id = job_ids[trajectory]
            prompt_stream = prompt_streams[trajectory]
            active_prompt_stream = prompt_stream
            previous_input_target: int | None = None
            context_segment = 0
            previous_tool: str | None = None

            for turn_index, event in enumerate(items):
                if abort_event.is_set():
                    return local_requests, local_events

                target_tokens = int(event["input_tokens_target"])

                if (
                    args.reset_aware_prefix
                    and previous_input_target is not None
                    and target_tokens < previous_input_target
                ):
                    context_segment += 1
                    remaining_max = max(
                        int(item["input_tokens_target"])
                        for item in items[turn_index:]
                    )
                    segment_identity = (
                        trajectory_index
                        + context_segment * max(1, len(trajectories))
                    )
                    active_prompt_stream = build_prompt_tokens(
                        tokenizer,
                        segment_identity,
                        remaining_max,
                        args.common_prefix_tokens,
                    )

                request_id = f"envbench-{event['global_event_index']}"
                request_start_offset = (
                    time.perf_counter() - start_perf
                )
                result = stream_request(
                    completions_url, args.model,
                    active_prompt_stream[:target_tokens],
                    args.max_output_tokens, job_id,
                    previous_tool, event["tool_name"], False,
                    request_id, tokenizer, args.request_timeout,
                )
                local_requests.append({
                    "request_sequence": None,
                    "request_start_offset_seconds": request_start_offset,
                    "worker_index": worker_index,
                    "request_kind": "tool_event",
                    "request_id_client": request_id,
                    "job_id": job_id,
                    "trajectory": trajectory,
                    "trajectory_index": trajectory_index,
                    "turn_index": turn_index,
                    "context_segment": context_segment,
                    "tool_name": event["tool_name"],
                    **result,
                })

                if not result["success"]:
                    abort_event.set()
                    raise RuntimeError(
                        "Runtime replay aborted after request failure: "
                        f"worker={worker_index}, job={job_id}, "
                        f"trajectory={trajectory}, turn={turn_index}, "
                        f"request={request_id}, "
                        f"status={result.get('status_code')}, "
                        f"error={result.get('error')}"
                    )

                scaled = (
                    event["duration_seconds_raw"]
                    * args.duration_scale
                )
                effective = min(
                    scaled,
                    args.max_sleep_seconds,
                )
                local_events.append({
                    **event,
                    "job_id": job_id,
                    "worker_index": worker_index,
                    "trajectory_index": trajectory_index,
                    "turn_index": turn_index,
                    "context_segment": context_segment,
                    "request_start_offset_seconds": (
                        request_start_offset
                    ),
                    "request_success": result["success"],
                    "duration_scale": args.duration_scale,
                    "max_sleep_seconds": args.max_sleep_seconds,
                    "duration_seconds_effective": effective,
                    "duration_was_capped": int(
                        scaled > args.max_sleep_seconds
                    ),
                })

                previous_input_target = target_tokens

                if effective > 0.0 and abort_event.wait(effective):
                    return local_requests, local_events
                previous_tool = event["tool_name"]

            if items and not abort_event.is_set():
                last = items[-1]
                terminal_id = (
                    f"envbench-terminal-{trajectory_index:05d}"
                )
                request_start_offset = (
                    time.perf_counter() - start_perf
                )
                terminal_result = stream_request(
                    completions_url, args.model,
                    active_prompt_stream[: int(last["input_tokens_target"])],
                    1, job_id, previous_tool, None, True,
                    terminal_id, tokenizer, args.request_timeout,
                )
                local_requests.append({
                    "request_sequence": None,
                    "request_start_offset_seconds": request_start_offset,
                    "worker_index": worker_index,
                    "request_kind": "terminal",
                    "request_id_client": terminal_id,
                    "job_id": job_id,
                    "trajectory": trajectory,
                    "trajectory_index": trajectory_index,
                    "turn_index": len(items),
                    "context_segment": context_segment,
                    "tool_name": None,
                    **terminal_result,
                })
                if not terminal_result["success"]:
                    abort_event.set()
                    raise RuntimeError(
                        "Runtime replay aborted after terminal request "
                        "failure: "
                        f"worker={worker_index}, job={job_id}, "
                        f"trajectory={trajectory}, "
                        f"request={terminal_id}, "
                        f"status={terminal_result.get('status_code')}, "
                        f"error={terminal_result.get('error')}"
                    )

        return local_requests, local_events

    poller.start()
    try:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="envbench-worker",
        ) as executor:
            futures = [
                executor.submit(run_lane, worker_index, lane)
                for worker_index, lane in enumerate(lanes)
            ]
            start_event.set()
            for future in as_completed(futures):
                try:
                    lane_requests, lane_events = future.result()
                except Exception:
                    abort_event.set()
                    for pending in futures:
                        pending.cancel()
                    raise
                request_rows.extend(lane_requests)
                event_rows.extend(lane_events)
    finally:
        abort_event.set()
        start_event.set()
        poller.stop()

    request_rows.sort(
        key=lambda row: (
            row["request_start_offset_seconds"],
            row["trajectory_index"],
            row["turn_index"],
        )
    )
    for sequence, row in enumerate(request_rows):
        row["request_sequence"] = sequence

    event_rows.sort(
        key=lambda row: row["global_event_index"]
    )

    end_perf = time.perf_counter()
    end_wall = time.time()
    elapsed = end_perf - start_perf

    if args.log_flush_wait:
        time.sleep(args.log_flush_wait)

    metrics_after = fetch_text(metrics_url)
    (output_dir / "metrics_after.txt").write_text(
        metrics_after, encoding="utf-8"
    )
    with server_log.open("rb") as f:
        f.seek(min(log_offset, server_log.stat().st_size))
        log_append = f.read().decode("utf-8", errors="replace")
    (output_dir / "server_log_append.txt").write_text(
        log_append, encoding="utf-8"
    )

    ttl_rows = parse_ttl(log_append)
    attach_ttl(event_rows, ttl_rows)
    write_csv(output_dir / "requests.csv", request_rows)
    write_csv(output_dir / "tool_events.csv", event_rows)
    write_csv(output_dir / "kv_cache_samples.csv", poller.rows)
    write_csv(output_dir / "ttl_log_rows.csv", ttl_rows)

    total_requests = len(request_rows)
    success = sum(row["success"] for row in request_rows)
    failed = total_requests - success
    input_tokens = sum(
        row["prompt_tokens"] for row in request_rows if row["success"]
    )
    output_tokens = sum(
        row["completion_tokens"] for row in request_rows if row["success"]
    )
    total_tokens = input_tokens + output_tokens

    prefix_queries = delta(
        metric_sum(metrics_after, "vllm:prefix_cache_queries_total"),
        metric_sum(metrics_before, "vllm:prefix_cache_queries_total"),
    )
    prefix_hits = delta(
        metric_sum(metrics_after, "vllm:prefix_cache_hits_total"),
        metric_sum(metrics_before, "vllm:prefix_cache_hits_total"),
    )
    preemptions = delta(
        metric_sum(metrics_after, "vllm:num_preemptions_total"),
        metric_sum(metrics_before, "vllm:num_preemptions_total"),
    )

    matched = [row for row in event_rows if row.get("ttl_hit") is not None]
    ttl_hits = sum(row["ttl_hit"] for row in matched)
    ttl_timeouts = sum(row["ttl_timeout"] for row in matched)
    sources = Counter(
        row["ttl_history_source"] for row in matched
        if row.get("ttl_history_source")
    )

    summary = {
        "experiment": {
            "type": "EnvBench-derived vLLM-Continuum runtime",
            "original_prompts_reconstructed": False,
            "prompt_method": (
                "deterministic token-id prefixes matching input_tokens"
            ),
            "input_csv": str(input_path),
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "start_unix": start_wall,
            "end_unix": end_wall,
            "elapsed_seconds": elapsed,
            "selected_tool_events": len(event_rows),
            "trajectories": len(trajectories),
            "concurrency": worker_count,
            "lane_assignment": args.lane_assignment,
            "requested_concurrency": args.concurrency,
            "worker_assignment": "trajectory_index modulo concurrency",
            "experiment_label": args.experiment_label,
            "duration_scale": args.duration_scale,
            "max_sleep_seconds": args.max_sleep_seconds,
            "ttl_comparison_uses_effective_duration": True,
            "reset_aware_prefix": bool(args.reset_aware_prefix),
            "context_reset_policy": (
                "new deterministic token stream after input-token decrease"
                if args.reset_aware_prefix
                else "legacy single deterministic stream per trajectory"
            ),
        },
        "requests": {
            "total": total_requests,
            "success": success,
            "failed": failed,
            "tool_event_requests": len(event_rows),
            "terminal_requests": total_requests - len(event_rows),
            "throughput_success_req_per_s": (
                success / elapsed if elapsed else None
            ),
            "attempted_req_per_s": (
                total_requests / elapsed if elapsed else None
            ),
        },
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
            "input_throughput_tok_per_s": (
                input_tokens / elapsed if elapsed else None
            ),
            "output_throughput_tok_per_s": (
                output_tokens / elapsed if elapsed else None
            ),
            "total_throughput_tok_per_s": (
                total_tokens / elapsed if elapsed else None
            ),
        },
        "latency_seconds": {
            "ttft": distribution(
                row["ttft_seconds"] for row in request_rows
                if row["success"]
            ),
            "tpot": distribution(
                row["tpot_seconds"] for row in request_rows
                if row["success"]
            ),
            "e2e": distribution(
                row["e2e_seconds"] for row in request_rows
                if row["success"]
            ),
        },
        "prefix_cache": {
            "query_tokens": prefix_queries,
            "hit_tokens": prefix_hits,
            "token_hit_rate": (
                prefix_hits / prefix_queries if prefix_queries else None
            ),
        },
        "kv_cache_usage_fraction": distribution(
            row["kv_cache_usage_perc"] for row in poller.rows
        ),
        "preemptions": preemptions,
        "ttl": {
            "matched_events": len(matched),
            "unmatched_events": len(event_rows) - len(matched),
            "seconds": distribution(
                row["ttl_seconds"] for row in matched
            ),
            "hit_count": ttl_hits,
            "timeout_count": ttl_timeouts,
            "hit_rate": ttl_hits / len(matched) if matched else None,
            "timeout_rate": (
                ttl_timeouts / len(matched) if matched else None
            ),
            "history_source_counts": dict(sorted(sources.items())),
            "comparison_duration": "scaled and capped effective tool wait",
        },
        "server_metrics_delta": {
            "request_success_total": delta(
                metric_sum(metrics_after, "vllm:request_success_total"),
                metric_sum(metrics_before, "vllm:request_success_total"),
            ),
            "prompt_tokens_total": delta(
                metric_sum(metrics_after, "vllm:prompt_tokens_total"),
                metric_sum(metrics_before, "vllm:prompt_tokens_total"),
            ),
            "generation_tokens_total": delta(
                metric_sum(metrics_after, "vllm:generation_tokens_total"),
                metric_sum(metrics_before, "vllm:generation_tokens_total"),
            ),
        },
        "metrics_poller": {
            "samples": len(poller.rows),
            "errors": poller.errors,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("EnvBench-derived runtime test complete")
    print(f"Output directory:     {output_dir}")
    print(
        f"Requests:             {total_requests} total, "
        f"{success} success, {failed} failed"
    )
    print(f"Tool events:          {len(event_rows)}")
    print(f"Concurrency:          {worker_count}")
    print(f"Lane assignment:      {args.lane_assignment}")
    print(
        "Request throughput:   "
        f"{summary['requests']['throughput_success_req_per_s']:.4f} req/s"
    )
    print(
        "Total token throughput: "
        f"{summary['tokens']['total_throughput_tok_per_s']:.2f} tok/s"
    )
    print(
        f"Prefix token hit rate: {summary['prefix_cache']['token_hit_rate']}"
    )
    print(f"Preemptions:          {preemptions}")
    print(f"TTL matched:          {len(matched)}/{len(event_rows)}")
    print(f"TTL hit rate:         {summary['ttl']['hit_rate']}")
    print(f"Summary JSON:         {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final saturation EnvBench saturation runner with a shared HTTPX pool.

This file is intentionally additive. Put it next to the frozen final benchmark
scripts:

    benchmark/scripts/run_envbench_runtime_saturation.py

It reuses the mature prompt construction, metrics parsing, TTL parsing,
and result schemas from the frozen final concurrent runner. The scheduling
layer is a global ready-queue open-loop dispatcher, while inference HTTP
transport uses one thread-safe shared httpx.Client connection pool.

The material scheduling semantics are:

* each trajectory is a causal state machine;
* a trajectory's next turn becomes ready only after the previous request
  completes plus its simulated tool delay;
* globally ready requests are admitted at a deterministic fixed rate R;
* request completion does not gate admission of requests from other
  trajectories;
* max_inflight is only a safety fuse and is explicitly reported;
* the shared connection pool has an independent bounded FD footprint;
* no trajectory can reserve future admission slots before it becomes ready.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import time
import resource
import httpx
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import run_envbench_runtime_concurrent as base




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--max-events", type=int, default=1_000_000_000)
    p.add_argument("--min-input-tokens", type=int, default=0)
    p.add_argument("--max-input-tokens", type=int, default=4000)
    p.add_argument("--max-output-tokens", type=int, default=8)
    p.add_argument("--duration-scale", type=float, default=0.01)
    p.add_argument("--max-sleep-seconds", type=float, default=2.0)
    p.add_argument("--metrics-interval", type=float, default=0.2)
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--common-prefix-tokens", type=int, default=256)
    p.add_argument("--reset-aware-prefix", action="store_true")
    p.add_argument(
        "--target-request-rate",
        type=float,
        required=True,
        help="Global deterministic admission rate in requests/second.",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=512,
        help=(
            "Client safety fuse only. A valid saturation run should report "
            "peak_inflight < max_inflight and inflight_fuse_waits = 0."
        ),
    )
    p.add_argument(
        "--max-connections",
        type=int,
        default=512,
        help=(
            "Maximum simultaneous HTTP connections in the shared httpx pool. "
            "Keep this below the process FD soft limit with safety headroom."
        ),
    )
    p.add_argument(
        "--max-keepalive-connections",
        type=int,
        default=128,
        help="Maximum idle keep-alive connections retained by httpx.",
    )
    p.add_argument(
        "--keepalive-expiry",
        type=float,
        default=5.0,
        help="Idle keep-alive expiry in seconds.",
    )
    p.add_argument("--max-admissions", type=int, default=0, help="Stop admitting after N total requests; 0 means unlimited.")
    p.add_argument("--load-duration-seconds", type=float, default=0.0, help="Stop admitting after this many seconds; 0 means unlimited.")
    p.add_argument("--experiment-label", default="openloop")
    p.add_argument("--log-flush-wait", type=float, default=0.5)
    args = p.parse_args()

    if args.max_events < 1:
        raise ValueError("--max-events must be at least 1")
    if args.duration_scale < 0 or args.max_sleep_seconds < 0:
        raise ValueError("duration controls must be non-negative")
    if args.target_request_rate <= 0:
        raise ValueError("--target-request-rate must be > 0")
    if args.max_admissions < 0:
        raise ValueError("--max-admissions must be >= 0")
    if args.load_duration_seconds < 0:
        raise ValueError("--load-duration-seconds must be >= 0")
    if args.max_inflight < 1:
        raise ValueError("--max-inflight must be at least 1")
    if args.max_connections < 1:
        raise ValueError("--max-connections must be at least 1")
    if args.max_keepalive_connections < 0:
        raise ValueError("--max-keepalive-connections must be non-negative")
    if args.max_keepalive_connections > args.max_connections:
        raise ValueError(
            "--max-keepalive-connections cannot exceed --max-connections"
        )
    if args.max_connections < args.max_inflight:
        raise ValueError(
            "--max-connections must be >= --max-inflight so the HTTP pool "
            "cannot silently become the client concurrency ceiling"
        )
    if args.keepalive_expiry < 0:
        raise ValueError("--keepalive-expiry must be non-negative")
    if args.log_flush_wait < 0:
        raise ValueError("--log-flush-wait must be non-negative")

    fd_soft, fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    fd_headroom = 128
    if fd_soft != resource.RLIM_INFINITY:
        safe_connections = max(1, int(fd_soft) - fd_headroom)
        if args.max_connections > safe_connections:
            raise ValueError(
                f"--max-connections={args.max_connections} is unsafe for "
                f"RLIMIT_NOFILE soft={fd_soft}; keep at least "
                f"{fd_headroom} descriptors of headroom"
            )
    args.fd_soft_limit = fd_soft
    args.fd_hard_limit = fd_hard
    args.fd_headroom = fd_headroom
    return args


def configure_http(args: argparse.Namespace):
    """Install a thread-safe shared HTTPX streaming transport.

    The final concurrent runner's prompt/response semantics are preserved,
    including SSE streaming, TTFT/TPOT/E2E stage fields, request metadata, and
    token accounting.  Only inference transport is replaced; /metrics polling
    remains on the frozen runner's original urllib path.
    """

    original_stream_request = base.stream_request

    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
        keepalive_expiry=args.keepalive_expiry,
    )
    client = httpx.Client(
        limits=limits,
        http2=False,
        trust_env=False,
    )

    def stream_request_httpx(
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
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-Id": request_id,
        }

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

            response_open_seconds = elapsed(response_open)
            return {
                "request_start_unix_seconds": start_wall,
                "response_open_unix_seconds": wall_time(response_open),
                "first_raw_line_unix_seconds": wall_time(first_raw_line),
                "first_sse_data_unix_seconds": wall_time(first_sse_data),
                "first_choice_unix_seconds": wall_time(first_choice),
                "first_token_unix_seconds": wall_time(first_token),
                "request_end_unix_seconds": wall_time(end_time),
                "response_open_seconds": response_open_seconds,
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
                # HTTPX owns the shared pool, so per-request connection-new/
                # reused and connect/send split are intentionally not guessed.
                "http_connection_reused": None,
                "http_connect_seconds": None,
                "http_send_seconds": None,
                "http_wait_headers_seconds": response_open_seconds,
            }

        parts: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        status_code: int | None = None

        try:
            body = json.dumps(payload).encode("utf-8")
            with client.stream(
                "POST",
                url,
                content=body,
                headers=headers,
                timeout=timeout,
            ) as response:
                response_open = time.perf_counter()
                status_code = int(response.status_code)

                if status_code >= 400:
                    error_body = response.read().decode(
                        "utf-8", errors="replace"
                    )
                    end = time.perf_counter()
                    return {
                        "success": 0,
                        "status_code": status_code,
                        "error": f"HTTP {status_code}: {error_body}",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "ttft_seconds": None,
                        "tpot_seconds": None,
                        "e2e_seconds": end - start,
                        "finish_reason": None,
                        **stage_fields(end),
                    }

                for line in response.iter_lines():
                    raw_time = time.perf_counter()
                    if first_raw_line is None:
                        first_raw_line = raw_time

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
                        piece = choice.get("text") or ""
                        if piece:
                            if first_token is None:
                                first_token = time.perf_counter()
                            parts.append(piece)

                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])

                end = time.perf_counter()

        except Exception as exc:  # noqa: BLE001
            end = time.perf_counter()
            return {
                "success": 0,
                "status_code": status_code,
                "error": f"{type(exc).__name__}: {exc}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "ttft_seconds": None,
                "tpot_seconds": None,
                "e2e_seconds": end - start,
                "finish_reason": None,
                **stage_fields(end),
            }

        output_text = "".join(parts)
        prompt_tokens = int(
            usage.get("prompt_tokens") or len(prompt_ids)
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            or len(
                tokenizer.encode(
                    output_text,
                    add_special_tokens=False,
                )
            )
        )
        total_tokens = int(
            usage.get("total_tokens")
            or prompt_tokens + completion_tokens
        )

        ttft = (
            first_token - start
            if first_token is not None
            else None
        )
        e2e = end - start
        tpot = None
        if first_token is not None and completion_tokens > 1:
            tpot = (
                end - first_token
            ) / (completion_tokens - 1)

        return {
            "success": 1,
            "status_code": status_code,
            "error": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "ttft_seconds": ttft,
            "tpot_seconds": tpot,
            "e2e_seconds": e2e,
            "finish_reason": finish_reason,
            **stage_fields(end),
        }

    base.stream_request = stream_request_httpx

    def restore() -> None:
        try:
            client.close()
        finally:
            base.stream_request = original_stream_request

    return restore


def main() -> None:
    args = parse_args()
    restore_http = configure_http(args)

    try:
        root = Path.cwd()
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path

        run_dir = Path(args.run_dir)
        server_log = run_dir / "server.log"
        if not server_log.exists():
            raise FileNotFoundError(f"Missing server log: {server_log}")

        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.experiment_label)
        output_dir = run_dir / (
            f"envbench_saturation_{safe_label}_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        rows = base.load_rows(
            input_path,
            args.max_events,
            args.min_input_tokens,
            args.max_input_tokens,
        )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        first_index: dict[str, int] = {}
        for row in rows:
            grouped[row["trajectory"]].append(row)
            first_index.setdefault(row["trajectory"], row["global_event_index"])
        trajectories = sorted(grouped, key=lambda x: first_index[x])

        tokenizer = base.AutoTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=True,
        )

        prompt_streams: dict[str, list[int]] = {}
        job_ids: dict[str, str] = {}
        for i, trajectory in enumerate(trajectories):
            max_target = max(
                row["input_tokens_target"] for row in grouped[trajectory]
            )
            prompt_streams[trajectory] = base.build_prompt_tokens(
                tokenizer,
                i,
                max_target,
                args.common_prefix_tokens,
            )
            job_ids[trajectory] = f"envbench-runtime-{i:05d}"

        completions_url = args.base_url.rstrip("/") + "/v1/completions"
        metrics_url = args.base_url.rstrip("/") + "/metrics"

        metrics_before = base.fetch_text(metrics_url)
        (output_dir / "metrics_before.txt").write_text(
            metrics_before,
            encoding="utf-8",
        )
        log_offset = server_log.stat().st_size

        request_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        poller = base.MetricsPoller(metrics_url, args.metrics_interval)

        start_wall = time.time()
        start_perf = time.perf_counter()
        interval = 1.0 / args.target_request_rate
        load_deadline = (start_perf + args.load_duration_seconds if args.load_duration_seconds > 0 else None)
        admission_stop_reason = None

        states: list[dict[str, Any]] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            items = sorted(
                grouped[trajectory],
                key=lambda x: x["trajectory_event_index"],
            )
            states.append(
                {
                    "trajectory_index": trajectory_index,
                    "trajectory": trajectory,
                    "items": items,
                    "job_id": job_ids[trajectory],
                    "active_prompt_stream": prompt_streams[trajectory],
                    "previous_input_target": None,
                    "context_segment": 0,
                    "previous_tool": None,
                    "turn_index": 0,
                    "done": False,
                }
            )

        # (ready_perf, insertion_sequence, trajectory_index)
        ready_heap: list[tuple[float, int, int]] = []
        heap_sequence = 0
        for state in states:
            heapq.heappush(
                ready_heap,
                (start_perf, heap_sequence, state["trajectory_index"]),
            )
            heap_sequence += 1

        futures: dict[Any, dict[str, Any]] = {}
        inflight_events: list[tuple[float, int]] = []
        actual_starts: list[float] = []
        completions: list[float] = []

        next_slot = start_perf
        max_ready_queue = 0
        max_inflight_submit = 0
        fuse_waits = 0
        admission_slots = 0
        ready_starvation_events = 0
        ready_starvation_seconds = 0.0

        def execute(meta: dict[str, Any]):
            actual_start = time.perf_counter()
            result = base.stream_request(
                completions_url,
                args.model,
                meta["prompt_tokens"],
                meta["max_output_tokens"],
                meta["job_id"],
                meta["previous_tool"],
                meta["current_tool"],
                meta["terminal"],
                meta["request_id"],
                tokenizer,
                args.request_timeout,
            )
            completed = time.perf_counter()
            return meta, result, actual_start, completed

        def harvest(done_futures) -> None:
            nonlocal heap_sequence

            for future in done_futures:
                futures.pop(future, None)
                meta, result, actual_start, completed = future.result()

                actual_starts.append(actual_start)
                completions.append(completed)
                inflight_events.append((actual_start, +1))
                inflight_events.append((completed, -1))

                state = states[meta["trajectory_index"]]
                request_start_offset = actual_start - start_perf

                request_rows.append(
                    {
                        "request_sequence": None,
                        "request_start_offset_seconds": request_start_offset,
                        "ready_offset_seconds": meta["ready_perf"] - start_perf,
                        "scheduled_admission_offset_seconds": (
                            meta["scheduled_perf"] - start_perf
                        ),
                        "admission_submit_offset_seconds": (
                            meta["submit_perf"] - start_perf
                        ),
                        "actual_admission_offset_seconds": request_start_offset,
                        "admission_lag_seconds": max(
                            0.0,
                            actual_start - meta["scheduled_perf"],
                        ),
                        "client_queue_delay_seconds": max(
                            0.0,
                            actual_start - meta["submit_perf"],
                        ),
                        "completion_offset_seconds": completed - start_perf,
                        "inflight_at_submit": meta["inflight_at_submit"],
                        "ready_queue_depth_at_submit": (
                            meta["ready_queue_depth_at_submit"]
                        ),
                        "worker_index": None,
                        "request_kind": meta["request_kind"],
                        "request_id_client": meta["request_id"],
                        "job_id": meta["job_id"],
                        "trajectory": meta["trajectory"],
                        "trajectory_index": meta["trajectory_index"],
                        "turn_index": meta["turn_index"],
                        "context_segment": meta["context_segment"],
                        "tool_name": meta["current_tool"],
                        **result,
                    }
                )

                if not result["success"]:
                    raise RuntimeError(
                        "Open-loop replay request failed: "
                        f"trajectory={meta['trajectory']} "
                        f"turn={meta['turn_index']} "
                        f"request={meta['request_id']} "
                        f"status={result.get('status_code')} "
                        f"error={result.get('error')}"
                    )

                if meta["request_kind"] == "tool_event":
                    event = meta["event"]
                    event_rows.append(
                        {
                            **event,
                            "job_id": meta["job_id"],
                            "worker_index": None,
                            "trajectory_index": meta["trajectory_index"],
                            "turn_index": meta["turn_index"],
                            "context_segment": meta["context_segment"],
                            "request_start_offset_seconds": request_start_offset,
                            "request_success": result["success"],
                            "duration_scale": args.duration_scale,
                            "max_sleep_seconds": args.max_sleep_seconds,
                            "duration_seconds_effective": meta["effective_wait"],
                            "duration_was_capped": int(
                                meta["scaled_wait"] > args.max_sleep_seconds
                            ),
                        }
                    )

                    state["previous_input_target"] = meta["target_tokens"]
                    state["previous_tool"] = meta["current_tool"]
                    state["turn_index"] += 1

                    next_ready = completed + meta["effective_wait"]
                    heapq.heappush(
                        ready_heap,
                        (
                            next_ready,
                            heap_sequence,
                            meta["trajectory_index"],
                        ),
                    )
                    heap_sequence += 1
                else:
                    state["done"] = True

        poller.start()
        try:
            with ThreadPoolExecutor(
                max_workers=args.max_inflight,
                thread_name_prefix="envbench-openloop",
            ) as executor:
                while ready_heap or futures:
                    now_loop = time.perf_counter()
                    hit_count_limit = args.max_admissions > 0 and admission_slots >= args.max_admissions
                    hit_time_limit = load_deadline is not None and now_loop >= load_deadline
                    if hit_count_limit or hit_time_limit:
                        if admission_stop_reason is None:
                            admission_stop_reason = "max_admissions" if hit_count_limit else "load_duration"
                        ready_heap.clear()
                        if futures:
                            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                            harvest(done)
                            continue
                        break
                    already_done = {f for f in list(futures) if f.done()}
                    if already_done:
                        harvest(already_done)
                        continue

                    if not ready_heap:
                        done, _ = wait(
                            set(futures),
                            return_when=FIRST_COMPLETED,
                        )
                        harvest(done)
                        continue

                    ready_perf, _, trajectory_index = ready_heap[0]
                    scheduled_perf = max(next_slot, ready_perf)

                    # If no request is ready at the next nominal slot, the
                    # workload's causal/tool timing, not the server, limits the
                    # offered rate. Record that rather than hiding it.
                    if ready_perf > next_slot:
                        ready_starvation_events += 1
                        ready_starvation_seconds += ready_perf - next_slot

                    if len(futures) >= args.max_inflight:
                        fuse_waits += 1
                        done, _ = wait(
                            set(futures),
                            return_when=FIRST_COMPLETED,
                        )
                        harvest(done)
                        continue

                    now = time.perf_counter()
                    wait_seconds = scheduled_perf - now
                    if wait_seconds > 0:
                        if futures:
                            done, _ = wait(
                                set(futures),
                                timeout=wait_seconds,
                                return_when=FIRST_COMPLETED,
                            )
                            if done:
                                harvest(done)
                                continue
                        else:
                            time.sleep(wait_seconds)
                        continue

                    heapq.heappop(ready_heap)
                    state = states[trajectory_index]
                    if state["done"]:
                        continue

                    turn_index = state["turn_index"]
                    items = state["items"]

                    if turn_index < len(items):
                        event = items[turn_index]
                        target_tokens = int(event["input_tokens_target"])

                        if (
                            args.reset_aware_prefix
                            and state["previous_input_target"] is not None
                            and target_tokens < state["previous_input_target"]
                        ):
                            state["context_segment"] += 1
                            remaining_max = max(
                                int(item["input_tokens_target"])
                                for item in items[turn_index:]
                            )
                            segment_identity = (
                                trajectory_index
                                + state["context_segment"]
                                * max(1, len(trajectories))
                            )
                            state["active_prompt_stream"] = (
                                base.build_prompt_tokens(
                                    tokenizer,
                                    segment_identity,
                                    remaining_max,
                                    args.common_prefix_tokens,
                                )
                            )

                        scaled_wait = (
                            event["duration_seconds_raw"] * args.duration_scale
                        )
                        effective_wait = min(
                            scaled_wait,
                            args.max_sleep_seconds,
                        )

                        meta = {
                            "request_kind": "tool_event",
                            "terminal": False,
                            "request_id": (
                                f"envbench-{event['global_event_index']}"
                            ),
                            "trajectory_index": trajectory_index,
                            "trajectory": state["trajectory"],
                            "job_id": state["job_id"],
                            "turn_index": turn_index,
                            "context_segment": state["context_segment"],
                            "previous_tool": state["previous_tool"],
                            "current_tool": event["tool_name"],
                            "target_tokens": target_tokens,
                            "prompt_tokens": state["active_prompt_stream"][
                                :target_tokens
                            ],
                            "max_output_tokens": args.max_output_tokens,
                            "event": event,
                            "scaled_wait": scaled_wait,
                            "effective_wait": effective_wait,
                        }
                    else:
                        if not items:
                            state["done"] = True
                            continue

                        last = items[-1]
                        target_tokens = int(last["input_tokens_target"])
                        meta = {
                            "request_kind": "terminal",
                            "terminal": True,
                            "request_id": (
                                f"envbench-terminal-{trajectory_index:05d}"
                            ),
                            "trajectory_index": trajectory_index,
                            "trajectory": state["trajectory"],
                            "job_id": state["job_id"],
                            "turn_index": len(items),
                            "context_segment": state["context_segment"],
                            "previous_tool": state["previous_tool"],
                            "current_tool": None,
                            "target_tokens": target_tokens,
                            "prompt_tokens": state["active_prompt_stream"][
                                :target_tokens
                            ],
                            "max_output_tokens": 1,
                            "event": None,
                            "scaled_wait": 0.0,
                            "effective_wait": 0.0,
                        }

                    submit_perf = time.perf_counter()
                    ready_depth = 1 + sum(
                        1 for item in ready_heap if item[0] <= submit_perf
                    )
                    max_ready_queue = max(max_ready_queue, ready_depth)

                    inflight_at_submit = len(futures) + 1
                    max_inflight_submit = max(
                        max_inflight_submit,
                        inflight_at_submit,
                    )

                    meta.update(
                        {
                            "ready_perf": ready_perf,
                            "scheduled_perf": scheduled_perf,
                            "submit_perf": submit_perf,
                            "inflight_at_submit": inflight_at_submit,
                            "ready_queue_depth_at_submit": ready_depth,
                        }
                    )

                    future = executor.submit(execute, meta)
                    futures[future] = meta
                    admission_slots += 1

                    # Advance on an absolute deterministic time grid.
                    # Dispatcher bookkeeping must not accumulate into the
                    # requested inter-arrival interval. If a whole slot is
                    # missed, skip it instead of generating a catch-up burst.
                    after_submit = time.perf_counter()
                    next_slot = scheduled_perf + interval
                    if after_submit - next_slot >= interval:
                        skipped = int((after_submit - next_slot) // interval)
                        next_slot += skipped * interval
        finally:
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

        event_rows.sort(key=lambda row: row["global_event_index"])

        end_perf = time.perf_counter()
        end_wall = time.time()
        elapsed = end_perf - start_perf

        if args.log_flush_wait:
            time.sleep(args.log_flush_wait)

        metrics_after = base.fetch_text(metrics_url)
        (output_dir / "metrics_after.txt").write_text(
            metrics_after,
            encoding="utf-8",
        )

        with server_log.open("rb") as f:
            f.seek(min(log_offset, server_log.stat().st_size))
            log_append = f.read().decode("utf-8", errors="replace")

        (output_dir / "server_log_append.txt").write_text(
            log_append,
            encoding="utf-8",
        )

        ttl_rows = base.parse_ttl(log_append)
        base.attach_ttl(event_rows, ttl_rows)

        base.write_csv(output_dir / "requests.csv", request_rows)
        base.write_csv(output_dir / "tool_events.csv", event_rows)
        base.write_csv(output_dir / "kv_cache_samples.csv", poller.rows)
        base.write_csv(output_dir / "ttl_log_rows.csv", ttl_rows)

        total_requests = len(request_rows)
        success = sum(row["success"] for row in request_rows)
        failed = total_requests - success

        input_tokens = sum(
            row["prompt_tokens"]
            for row in request_rows
            if row["success"]
        )
        output_tokens = sum(
            row["completion_tokens"]
            for row in request_rows
            if row["success"]
        )
        total_tokens = input_tokens + output_tokens

        prefix_queries = base.delta(
            base.metric_sum(metrics_after, "vllm:prefix_cache_queries_total"),
            base.metric_sum(metrics_before, "vllm:prefix_cache_queries_total"),
        )
        prefix_hits = base.delta(
            base.metric_sum(metrics_after, "vllm:prefix_cache_hits_total"),
            base.metric_sum(metrics_before, "vllm:prefix_cache_hits_total"),
        )
        preemptions = base.delta(
            base.metric_sum(metrics_after, "vllm:num_preemptions_total"),
            base.metric_sum(metrics_before, "vllm:num_preemptions_total"),
        )

        matched = [
            row for row in event_rows if row.get("ttl_hit") is not None
        ]
        ttl_hits = sum(row["ttl_hit"] for row in matched)
        ttl_timeouts = sum(row["ttl_timeout"] for row in matched)
        sources = Counter(
            row["ttl_history_source"]
            for row in matched
            if row.get("ttl_history_source")
        )

        if actual_starts:
            ordered_starts = sorted(actual_starts)
            first_start = ordered_starts[0]
            last_start = ordered_starts[-1]
            load_phase_seconds = max(
                interval,
                (last_start - first_start) + interval,
            )
            actual_arrival_rate = (
                len(ordered_starts) / load_phase_seconds
            )

            # Central-90% arrival rate is less sensitive to the finite-workload
            # startup/tail while still exposing sustained generator shortfall.
            trim = int(len(ordered_starts) * 0.05)
            central = (
                ordered_starts[trim:len(ordered_starts) - trim]
                if trim > 0 and len(ordered_starts) > 2 * trim + 1
                else ordered_starts
            )
            if len(central) >= 2:
                central_window = max(
                    interval,
                    (central[-1] - central[0]) + interval,
                )
                central_arrival_rate = (
                    len(central) / central_window
                )
            else:
                central_arrival_rate = actual_arrival_rate

            actual_to_target_ratio = (
                actual_arrival_rate / args.target_request_rate
            )

            arrival_intervals = [
                b - a
                for a, b in zip(
                    ordered_starts,
                    ordered_starts[1:],
                )
            ]

            last_completion = (
                max(completions) if completions else last_start
            )
            drain_seconds = max(
                0.0,
                last_completion - last_start,
            )
            completed_by_last_admission = sum(
                1 for t in completions if t <= last_start
            )
            steady_completed_rate = (
                completed_by_last_admission / load_phase_seconds
            )
        else:
            load_phase_seconds = 0.0
            actual_arrival_rate = None
            central_arrival_rate = None
            actual_to_target_ratio = None
            arrival_intervals = []
            drain_seconds = 0.0
            completed_by_last_admission = 0
            steady_completed_rate = None

        inflight = 0
        peak_inflight = 0
        inflight_area = 0.0
        cursor = start_perf

        for stamp, d in sorted(
            inflight_events,
            key=lambda item: (item[0], -item[1]),
        ):
            inflight_area += inflight * max(0.0, stamp - cursor)
            cursor = stamp
            inflight += d
            peak_inflight = max(peak_inflight, inflight)

        inflight_area += inflight * max(0.0, end_perf - cursor)
        mean_inflight = (
            inflight_area / elapsed if elapsed > 0 else None
        )

        summary = {
            "experiment": {
                "type": "EnvBench-derived open-loop saturation runtime",
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
                "experiment_label": args.experiment_label,
                "duration_scale": args.duration_scale,
                "max_sleep_seconds": args.max_sleep_seconds,
                "ttl_comparison_uses_effective_duration": True,
                "reset_aware_prefix": bool(args.reset_aware_prefix),
                "arrival_mode": "deterministic_fixed_interval",
                "http_transport": "httpx_shared_pool",
                "http_max_connections": args.max_connections,
                "http_max_keepalive_connections": (
                    args.max_keepalive_connections
                ),
                "http_keepalive_expiry_seconds": args.keepalive_expiry,
                "client_fd_soft_limit": args.fd_soft_limit,
                "client_fd_hard_limit": args.fd_hard_limit,
                "client_fd_reserved_headroom": args.fd_headroom,
            },
            "open_loop": {
                "target_request_rate_req_per_s": args.target_request_rate,
                "target_interarrival_seconds": interval,
                "configured_max_admissions": args.max_admissions,
                "configured_load_duration_seconds": args.load_duration_seconds,
                "admission_stop_reason": admission_stop_reason or "workload_exhausted",
                "actual_arrival_rate_req_per_s": actual_arrival_rate,
                "actual_arrival_rate_central90_req_per_s": (
                    central_arrival_rate
                ),
                "actual_to_target_ratio": actual_to_target_ratio,
                "actual_interarrival_seconds": base.distribution(
                    arrival_intervals
                ),
                "load_phase_seconds": load_phase_seconds,
                "admission_slots": admission_slots,
                "completed_by_last_admission": completed_by_last_admission,
                "steady_completed_req_per_s": steady_completed_rate,
                "drain_seconds": drain_seconds,
                "max_inflight_configured": args.max_inflight,
                "peak_inflight": peak_inflight,
                "peak_inflight_at_submit": max_inflight_submit,
                "mean_inflight_time_weighted": mean_inflight,
                "max_ready_queue": max_ready_queue,
                "inflight_fuse_waits": fuse_waits,
                "ready_starvation_events": ready_starvation_events,
                "ready_starvation_seconds": ready_starvation_seconds,
                "admission_lag_seconds": base.distribution(
                    row["admission_lag_seconds"] for row in request_rows
                ),
                "client_queue_delay_seconds": base.distribution(
                    row["client_queue_delay_seconds"] for row in request_rows
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
                "ttft": base.distribution(
                    row["ttft_seconds"]
                    for row in request_rows
                    if row["success"]
                ),
                "tpot": base.distribution(
                    row["tpot_seconds"]
                    for row in request_rows
                    if row["success"]
                ),
                "e2e": base.distribution(
                    row["e2e_seconds"]
                    for row in request_rows
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
            "kv_cache_usage_fraction": base.distribution(
                row["kv_cache_usage_perc"] for row in poller.rows
            ),
            "preemptions": preemptions,
            "ttl": {
                "matched_events": len(matched),
                "unmatched_events": len(event_rows) - len(matched),
                "seconds": base.distribution(
                    row["ttl_seconds"] for row in matched
                ),
                "hit_count": ttl_hits,
                "timeout_count": ttl_timeouts,
                "hit_rate": (
                    ttl_hits / len(matched) if matched else None
                ),
                "timeout_rate": (
                    ttl_timeouts / len(matched) if matched else None
                ),
                "history_source_counts": dict(sorted(sources.items())),
                "comparison_duration": (
                    "scaled and capped effective tool wait"
                ),
            },
            "server_metrics_delta": {
                "request_success_total": base.delta(
                    base.metric_sum(
                        metrics_after,
                        "vllm:request_success_total",
                    ),
                    base.metric_sum(
                        metrics_before,
                        "vllm:request_success_total",
                    ),
                ),
                "prompt_tokens_total": base.delta(
                    base.metric_sum(
                        metrics_after,
                        "vllm:prompt_tokens_total",
                    ),
                    base.metric_sum(
                        metrics_before,
                        "vllm:prompt_tokens_total",
                    ),
                ),
                "generation_tokens_total": base.delta(
                    base.metric_sum(
                        metrics_after,
                        "vllm:generation_tokens_total",
                    ),
                    base.metric_sum(
                        metrics_before,
                        "vllm:generation_tokens_total",
                    ),
                ),
            },
            "server_queue": {
                "running_requests": base.distribution(
                    row.get("num_requests_running")
                    for row in poller.rows
                ),
                "waiting_requests": base.distribution(
                    row.get("num_requests_waiting")
                    for row in poller.rows
                ),
            },
            "client_capacity_guard": {
                "configured_max_inflight": args.max_inflight,
                "configured_max_connections": args.max_connections,
                "peak_inflight": peak_inflight,
                "inflight_fuse_waits": fuse_waits,
                "ceiling_hit": (
                    fuse_waits > 0
                    or peak_inflight >= args.max_inflight
                    or peak_inflight >= args.max_connections
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

        print("EnvBench-derived SATURATION runtime test complete")
        print(f"Output directory:       {output_dir}")
        print(
            f"Requests:               {total_requests} total, "
            f"{success} success, {failed} failed"
        )
        print(f"Tool events:            {len(event_rows)}")
        print(
            "Target request rate:    "
            f"{args.target_request_rate:.4f} req/s"
        )
        print(f"Actual arrival rate:    {actual_arrival_rate}")
        print(f"Central90 arrival:      {central_arrival_rate}")
        print(f"Actual/target ratio:    {actual_to_target_ratio}")
        print(f"Steady completed rate:  {steady_completed_rate}")
        print(
            "Request throughput:     "
            f"{summary['requests']['throughput_success_req_per_s']:.4f} req/s"
        )
        print(
            f"Peak inflight:          {peak_inflight}/{args.max_inflight}"
        )
        print(f"Inflight fuse waits:    {fuse_waits}")
        print(
            f"HTTP pool:              active<= {args.max_connections}, "
            f"keepalive<= {args.max_keepalive_connections}"
        )
        print(
            f"Client FD limit:         {args.fd_soft_limit} "
            f"(reserved headroom {args.fd_headroom})"
        )
        print(f"Max ready queue:        {max_ready_queue}")
        print(f"Drain seconds:          {drain_seconds:.6f}")
        print(
            "Total token throughput: "
            f"{summary['tokens']['total_throughput_tok_per_s']:.2f} tok/s"
        )
        print(
            "Prefix token hit rate:  "
            f"{summary['prefix_cache']['token_hit_rate']}"
        )
        print(f"Preemptions:            {preemptions}")
        print(f"TTL matched:            {len(matched)}/{len(event_rows)}")
        print(f"TTL hit rate:           {summary['ttl']['hit_rate']}")
        print(f"Summary JSON:           {summary_path}")

    finally:
        if restore_http is not None:
            restore_http()


if __name__ == "__main__":
    main()

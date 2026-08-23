
#!/usr/bin/env python3

"""Build the full runtime-eligible EnvBench replay workload.



Selection policy:

- source: normalized EnvBench single-call trace

- preserve complete trajectories

- accept a trajectory only if every event is runtime-compatible

- do not pad, truncate, sample, score, or otherwise modify events

- context-length decreases are retained and audited as reset boundaries

"""



from __future__ import annotations



import argparse

import csv

import json

import math

import statistics

from collections import Counter, defaultdict

from pathlib import Path

from typing import Any





def quantile(values: list[float], q: float) -> float | None:

    if not values:

        return None

    xs = sorted(values)

    if len(xs) == 1:

        return float(xs[0])

    pos = (len(xs) - 1) * q

    lo = math.floor(pos)

    hi = math.ceil(pos)

    if lo == hi:

        return float(xs[lo])

    frac = pos - lo

    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)





def stats(values: list[float]) -> dict[str, Any]:

    if not values:

        return {

            "count": 0,

            "min": None,

            "mean": None,

            "median": None,

            "p90": None,

            "p95": None,

            "p99": None,

            "max": None,

        }

    return {

        "count": len(values),

        "min": min(values),

        "mean": statistics.fmean(values),

        "median": statistics.median(values),

        "p90": quantile(values, 0.90),

        "p95": quantile(values, 0.95),

        "p99": quantile(values, 0.99),

        "max": max(values),

    }





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)

    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--audit", type=Path, required=True)

    parser.add_argument("--min-input-tokens", type=int, default=1)

    parser.add_argument("--max-input-tokens", type=int, default=4000)

    return parser.parse_args()





def main() -> None:

    args = parse_args()



    with args.input.open("r", encoding="utf-8", newline="") as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:

            raise RuntimeError("Input CSV has no header")

        fieldnames = list(reader.fieldnames)

        rows = list(reader)



    required = {

        "global_event_index",

        "trajectory_event_index",

        "event_id",

        "trajectory",

        "environment",

        "agent_config",

        "tool_name",

        "duration_seconds",

        "input_tokens",

        "is_error",

    }

    missing = required.difference(fieldnames)

    if missing:

        raise ValueError(

            "Input CSV is missing columns: " + ", ".join(sorted(missing))

        )



    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:

        grouped[row["trajectory"]].append(row)



    accepted: dict[str, list[dict[str, str]]] = {}

    rejected_reason = Counter()



    for trajectory, items in grouped.items():

        items = sorted(

            items,

            key=lambda row: int(row["trajectory_event_index"]),

        )



        reason = None

        for row in items:

            token_text = row["input_tokens"].strip()

            duration_text = row["duration_seconds"].strip()



            if not token_text:

                reason = "missing_input_tokens"

                break

            if not duration_text:

                reason = "missing_duration"

                break



            try:

                tokens = int(float(token_text))

                duration = float(duration_text)

            except ValueError:

                reason = "invalid_numeric_value"

                break



            if not args.min_input_tokens <= tokens <= args.max_input_tokens:

                if tokens < args.min_input_tokens:

                    reason = "input_tokens_below_min"

                else:

                    reason = "input_tokens_above_max"

                break



            if duration < 0.0 or not math.isfinite(duration):

                reason = "invalid_duration"

                break



        if reason is not None:

            rejected_reason[reason] += 1

            continue



        accepted[trajectory] = items



    accepted_rows = [

        row

        for items in accepted.values()

        for row in items

    ]

    accepted_rows.sort(key=lambda row: int(row["global_event_index"]))



    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:

        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(accepted_rows)



    input_tokens = [

        int(float(row["input_tokens"]))

        for row in accepted_rows

    ]

    durations = [

        float(row["duration_seconds"])

        for row in accepted_rows

    ]



    events_per_trajectory = [

        len(items)

        for items in accepted.values()

    ]



    reset_trajectories = 0

    reset_events = 0

    reset_drop_tokens: list[float] = []

    reset_before_tokens: list[float] = []

    reset_after_tokens: list[float] = []



    for items in accepted.values():

        xs = [int(float(row["input_tokens"])) for row in items]

        has_reset = False

        for index in range(len(xs) - 1):

            if xs[index + 1] < xs[index]:

                has_reset = True

                reset_events += 1

                reset_drop_tokens.append(xs[index] - xs[index + 1])

                reset_before_tokens.append(xs[index])

                reset_after_tokens.append(xs[index + 1])

        if has_reset:

            reset_trajectories += 1



    environment_counts = Counter(

        items[0]["environment"]

        for items in accepted.values()

    )

    agent_config_counts = Counter(

        items[0]["agent_config"]

        for items in accepted.values()

    )

    tool_counts = Counter(

        row["tool_name"]

        for row in accepted_rows

    )



    token_bins = {

        "<=1000": sum(value <= 1000 for value in input_tokens),

        "1001-2000": sum(1000 < value <= 2000 for value in input_tokens),

        "2001-3500": sum(2000 < value <= 3500 for value in input_tokens),

        "3501-4000": sum(3500 < value <= 4000 for value in input_tokens),

    }



    audit = {

        "definition": (

            "Full runtime-eligible EnvBench trajectory workload under "

            "the frozen 4096-token serving configuration"

        ),

        "selection_policy": {

            "whole_trajectory_preserved": True,

            "min_input_tokens": args.min_input_tokens,

            "max_input_tokens": args.max_input_tokens,

            "pressure_padding": False,

            "high_contention_scoring": False,

            "sampling": False,

            "turn_truncation": False,

            "context_resets_preserved": True,

        },

        "source": {

            "path": str(args.input),

            "trajectories": len(grouped),

            "events": len(rows),

        },

        "accepted": {

            "trajectories": len(accepted),

            "events": len(accepted_rows),

            "expected_requests": len(accepted_rows) + len(accepted),

            "input_tokens_total": sum(input_tokens),

        },

        "rejected": {

            "trajectories": len(grouped) - len(accepted),

            "reasons": dict(sorted(rejected_reason.items())),

        },

        "input_tokens": stats([float(value) for value in input_tokens]),

        "input_token_bins": token_bins,

        "tool_duration_seconds": stats(durations),

        "events_per_trajectory": stats(

            [float(value) for value in events_per_trajectory]

        ),

        "context_resets": {

            "trajectories_with_reset": reset_trajectories,

            "trajectories_without_reset": len(accepted) - reset_trajectories,

            "reset_events": reset_events,

            "drop_tokens": stats(reset_drop_tokens),

            "before_reset_tokens": stats(reset_before_tokens),

            "after_reset_tokens": stats(reset_after_tokens),

        },

        "environment_trajectories": dict(

            sorted(environment_counts.items())

        ),

        "agent_config_trajectories": dict(

            sorted(agent_config_counts.items())

        ),

        "tool_events": dict(sorted(tool_counts.items())),

    }



    args.audit.parent.mkdir(parents=True, exist_ok=True)

    args.audit.write_text(

        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",

        encoding="utf-8",

    )



    print("FULL_ENVBench_BUILD=PASS")

    print(f"source_trajectories={len(grouped)}")

    print(f"source_events={len(rows)}")

    print(f"accepted_trajectories={len(accepted)}")

    print(f"accepted_events={len(accepted_rows)}")

    print(

        "expected_requests="

        f"{len(accepted_rows) + len(accepted)}"

    )

    print(f"input_tokens_total={sum(input_tokens)}")

    print(f"reset_trajectories={reset_trajectories}")

    print(f"reset_events={reset_events}")

    print(

        "rejected_trajectories="

        f"{len(grouped) - len(accepted)}"

    )

    print(f"output={args.output}")

    print(f"audit={args.audit}")





if __name__ == "__main__":

    main()


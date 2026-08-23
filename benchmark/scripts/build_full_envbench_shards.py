
#!/usr/bin/env python3

from __future__ import annotations



import argparse

import csv

import hashlib

import json

from collections import defaultdict

from pathlib import Path





def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(lambda: f.read(1024 * 1024), b""):

            h.update(chunk)

    return h.hexdigest()





def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", type=Path, required=True)

    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--shards", type=int, default=9)

    args = ap.parse_args()



    with args.input.open("r", encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:

            raise RuntimeError("missing CSV header")

        fieldnames = list(reader.fieldnames)

        rows = list(reader)



    grouped = defaultdict(list)

    for row in rows:

        grouped[row["trajectory"]].append(row)



    for items in grouped.values():

        items.sort(key=lambda r: int(r["trajectory_event_index"]))



    ntraj = len(grouped)

    base = ntraj // args.shards

    extra = ntraj % args.shards

    capacities = [

        base + (1 if i < extra else 0)

        for i in range(args.shards)

    ]



    trajectory_info = []

    for trajectory, items in grouped.items():

        tokens = sum(int(float(r["input_tokens"])) for r in items)

        events = len(items)

        resets = sum(

            int(float(items[i + 1]["input_tokens"]))

            < int(float(items[i]["input_tokens"]))

            for i in range(len(items) - 1)

        )

        first_index = min(int(r["global_event_index"]) for r in items)

        trajectory_info.append(

            {

                "trajectory": trajectory,

                "tokens": tokens,

                "events": events,

                "resets": resets,

                "first_index": first_index,

            }

        )



    # Deterministic largest-load-first balancing.

    trajectory_info.sort(

        key=lambda x: (

            -x["tokens"],

            -x["events"],

            x["trajectory"],

        )

    )



    shards = [

        {

            "trajectories": [],

            "tokens": 0,

            "events": 0,

            "resets": 0,

        }

        for _ in range(args.shards)

    ]



    for info in trajectory_info:

        eligible = [

            i for i in range(args.shards)

            if len(shards[i]["trajectories"]) < capacities[i]

        ]

        if not eligible:

            raise RuntimeError("no shard capacity remaining")



        index = min(

            eligible,

            key=lambda i: (

                shards[i]["tokens"],

                shards[i]["events"],

                len(shards[i]["trajectories"]),

                i,

            ),

        )

        shard = shards[index]

        shard["trajectories"].append(info["trajectory"])

        shard["tokens"] += info["tokens"]

        shard["events"] += info["events"]

        shard["resets"] += info["resets"]



    args.output_dir.mkdir(parents=True, exist_ok=True)



    manifest_rows = []

    all_written_event_ids = []



    for i, shard in enumerate(shards):

        selected = set(shard["trajectories"])



        out_rows = [

            row for row in rows

            if row["trajectory"] in selected

        ]

        out_rows.sort(key=lambda r: int(r["global_event_index"]))



        path = args.output_dir / f"shard_{i:02d}.csv"

        with path.open("w", encoding="utf-8", newline="") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            writer.writerows(out_rows)



        event_ids = [r["event_id"] for r in out_rows]

        all_written_event_ids.extend(event_ids)



        trajectory_count = len(selected)

        event_count = len(out_rows)

        expected_requests = trajectory_count + event_count

        token_total = sum(

            int(float(r["input_tokens"]))

            for r in out_rows

        )



        manifest_rows.append(

            {

                "shard": f"{i:02d}",

                "path": str(path),

                "trajectories": trajectory_count,

                "events": event_count,

                "expected_requests": expected_requests,

                "input_tokens": token_total,

                "reset_events": shard["resets"],

                "sha256": sha256(path),

            }

        )



    source_event_ids = [r["event_id"] for r in rows]

    if len(all_written_event_ids) != len(source_event_ids):

        raise RuntimeError("event count mismatch after sharding")

    if len(set(all_written_event_ids)) != len(all_written_event_ids):

        raise RuntimeError("duplicate event IDs across shards")

    if set(all_written_event_ids) != set(source_event_ids):

        raise RuntimeError("shards do not exactly cover source events")



    manifest = args.output_dir / "manifest.tsv"

    with manifest.open("w", encoding="utf-8", newline="") as f:

        fields = [

            "shard",

            "path",

            "trajectories",

            "events",

            "expected_requests",

            "input_tokens",

            "reset_events",

            "sha256",

        ]

        writer = csv.DictWriter(

            f,

            fieldnames=fields,

            delimiter="\t",

        )

        writer.writeheader()

        writer.writerows(manifest_rows)



    audit = {

        "input": str(args.input),

        "input_sha256": sha256(args.input),

        "shards": args.shards,

        "source": {

            "trajectories": ntraj,

            "events": len(rows),

            "input_tokens": sum(

                int(float(r["input_tokens"]))

                for r in rows

            ),

        },

        "capacities": capacities,

        "coverage_validation": "PASS",

        "manifest": manifest_rows,

    }



    audit_path = args.output_dir / "audit.json"

    audit_path.write_text(

        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",

        encoding="utf-8",

    )



    print("FULL_SHARD_BUILD=PASS")

    print(f"source_trajectories={ntraj}")

    print(f"source_events={len(rows)}")

    print(f"shards={args.shards}")

    print(f"capacities={capacities}")

    for row in manifest_rows:

        print(

            f"shard={row['shard']} "

            f"trajectories={row['trajectories']} "

            f"events={row['events']} "

            f"requests={row['expected_requests']} "

            f"tokens={row['input_tokens']} "

            f"resets={row['reset_events']}"

        )

    print("coverage_validation=PASS")

    print(f"manifest={manifest}")

    print(f"audit={audit_path}")





if __name__ == "__main__":

    main()


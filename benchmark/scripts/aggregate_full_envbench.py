
#!/usr/bin/env python3

from __future__ import annotations



import argparse

import csv

import json

import math

from collections import Counter, defaultdict

from pathlib import Path



import numpy as np





CASES = ["V", "U", "C", "F"]



CASE_NAMES = {

    "V": "vanilla_vllm",

    "U": "vllm_ucm",

    "C": "vllm_continuum",

    "F": "full_system",

}





def finite_float(x):

    if x is None:

        return None

    s = str(x).strip()

    if not s:

        return None

    try:

        v = float(s)

    except ValueError:

        return None

    return v if math.isfinite(v) else None





def truthy(x):

    return str(x).strip().lower() in {"1", "true", "yes", "y"}





def stats(values):

    a = np.asarray([float(x) for x in values if x is not None], dtype=float)

    a = a[np.isfinite(a)]

    if a.size == 0:

        return {

            "count": 0,

            "mean": None,

            "median": None,

            "p90": None,

            "p95": None,

            "p99": None,

            "max": None,

        }

    return {

        "count": int(a.size),

        "mean": float(np.mean(a)),

        "median": float(np.percentile(a, 50)),

        "p90": float(np.percentile(a, 90)),

        "p95": float(np.percentile(a, 95)),

        "p99": float(np.percentile(a, 99)),

        "max": float(np.max(a)),

    }





def fmt(x, digits=3):

    if x is None:

        return "N/A"

    return f"{x:.{digits}f}"





def pct(x, digits=2):

    if x is None:

        return "N/A"

    return f"{100.0 * x:.{digits}f}%"





def delta_pct(new, old):

    if new is None or old is None or old == 0:

        return None

    return (new / old - 1.0) * 100.0





def load_tsv(path):

    with path.open(encoding="utf-8", newline="") as f:

        return list(csv.DictReader(f, delimiter="\t"))





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", type=Path, required=True)

    ap.add_argument("--workload-manifest", type=Path, required=True)

    ap.add_argument("--output-dir", type=Path, required=True)

    args = ap.parse_args()



    rows = load_tsv(args.manifest)

    workloads = {

        r["shard"]: r

        for r in load_tsv(args.workload_manifest)

    }



    expected_pairs = {

        (f"{s:02d}", case)

        for s in range(9)

        for case in CASES

    }

    actual_pairs = {(r["shard"], r["case"]) for r in rows}



    assert len(rows) == 36

    assert actual_pairs == expected_pairs

    assert all(r["status"] == "PASS" for r in rows)



    args.output_dir.mkdir(parents=True, exist_ok=True)

    pooled_requests_dir = args.output_dir / "pooled_requests"

    pooled_jct_dir = args.output_dir / "pooled_jct"

    pooled_requests_dir.mkdir(exist_ok=True)

    pooled_jct_dir.mkdir(exist_ok=True)



    aggregate = {}



    for case in CASES:

        case_rows = sorted(

            [r for r in rows if r["case"] == case],

            key=lambda r: r["shard"],

        )



        total_elapsed = 0.0

        total_requests = 0

        total_success = 0

        total_failed = 0

        input_tokens = 0

        output_tokens = 0

        total_tokens = 0

        prefix_query = 0.0

        prefix_hit = 0.0

        preemptions = 0.0



        ttft_values = []

        tpot_values = []

        e2e_values = []

        jct_values = []

        ttl_seconds_values = []



        ttl_matched = 0

        ttl_unmatched = 0

        ttl_hit = 0

        ttl_timeout = 0

        ttl_history = Counter()



        kv_values = defaultdict(list)



        ucm_sums = Counter()

        ucm_reason_counts = Counter()

        ucm_event_counts = Counter()



        ssd_gib_values = []

        ssd_bytes_values = []



        request_writer = None

        request_fp = None

        request_fields = None



        jct_path = pooled_jct_dir / f"{case}.csv"

        jct_fp = jct_path.open("w", encoding="utf-8", newline="")

        jct_writer = csv.DictWriter(

            jct_fp,

            fieldnames=[

                "case",

                "shard",

                "job_id",

                "trajectory",

                "requests",

                "start_seconds",

                "end_seconds",

                "jct_seconds",

            ],

        )

        jct_writer.writeheader()



        pooled_request_path = pooled_requests_dir / f"{case}.csv"



        for manifest_row in case_rows:

            shard = manifest_row["shard"]

            run = Path(manifest_row["run_dir"])

            summary_path = Path(manifest_row["summary"])

            output_dir = summary_path.parent

            case_summary_path = run / "case_summary.json"



            assert summary_path.is_file(), summary_path

            assert case_summary_path.is_file(), case_summary_path



            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            case_summary = json.loads(

                case_summary_path.read_text(encoding="utf-8")

            )



            exp = summary["experiment"]

            req_summary = summary["requests"]

            tok_summary = summary["tokens"]

            prefix = summary["prefix_cache"]

            ttl_summary = summary["ttl"]



            expected_requests = int(

                workloads[shard]["expected_requests"]

            )



            assert exp["reset_aware_prefix"] is True

            assert exp["concurrency"] == 128

            assert exp["requested_concurrency"] == 128

            assert req_summary["total"] == expected_requests

            assert req_summary["success"] == expected_requests

            assert req_summary["failed"] == 0



            total_elapsed += float(exp["elapsed_seconds"])

            total_requests += int(req_summary["total"])

            total_success += int(req_summary["success"])

            total_failed += int(req_summary["failed"])



            input_tokens += int(tok_summary["input"])

            output_tokens += int(tok_summary["output"])

            total_tokens += int(tok_summary["total"])



            prefix_query += float(prefix["query_tokens"])

            prefix_hit += float(prefix["hit_tokens"])

            preemptions += float(summary.get("preemptions") or 0)



            ttl_matched += int(ttl_summary.get("matched_events") or 0)

            ttl_unmatched += int(ttl_summary.get("unmatched_events") or 0)

            ttl_hit += int(ttl_summary.get("hit_count") or 0)

            ttl_timeout += int(ttl_summary.get("timeout_count") or 0)

            ttl_history.update(

                ttl_summary.get("history_source_counts") or {}

            )



            requests_path = output_dir / "requests.csv"

            assert requests_path.is_file(), requests_path



            with requests_path.open(

                encoding="utf-8",

                newline="",

            ) as f:

                reader = csv.DictReader(f)

                fields = reader.fieldnames

                assert fields



                if request_fp is None:

                    request_fields = ["shard"] + fields

                    request_fp = pooled_request_path.open(

                        "w",

                        encoding="utf-8",

                        newline="",

                    )

                    request_writer = csv.DictWriter(

                        request_fp,

                        fieldnames=request_fields,

                    )

                    request_writer.writeheader()

                else:

                    assert fields == request_fields[1:]



                shard_request_rows = []

                for r in reader:

                    out = {"shard": shard}

                    out.update(r)

                    request_writer.writerow(out)



                    if not truthy(r["success"]):

                        continue



                    shard_request_rows.append(r)



                    v = finite_float(r["ttft_seconds"])

                    if v is not None:

                        ttft_values.append(v)



                    v = finite_float(r["tpot_seconds"])

                    if v is not None:

                        tpot_values.append(v)



                    v = finite_float(r["e2e_seconds"])

                    if v is not None:

                        e2e_values.append(v)



            by_job = defaultdict(list)

            for r in shard_request_rows:

                by_job[r["job_id"]].append(r)



            for job_id, job_rows in by_job.items():

                starts = [

                    finite_float(r["request_start_offset_seconds"])

                    for r in job_rows

                ]

                ends = []

                for r in job_rows:

                    s = finite_float(

                        r["request_start_offset_seconds"]

                    )

                    e = finite_float(r["e2e_seconds"])

                    if s is not None and e is not None:

                        ends.append(s + e)



                starts = [x for x in starts if x is not None]

                assert starts and ends



                start = min(starts)

                end = max(ends)

                jct = end - start

                jct_values.append(jct)



                trajectory = job_rows[0]["trajectory"]



                jct_writer.writerow(

                    {

                        "case": case,

                        "shard": shard,

                        "job_id": job_id,

                        "trajectory": trajectory,

                        "requests": len(job_rows),

                        "start_seconds": f"{start:.9f}",

                        "end_seconds": f"{end:.9f}",

                        "jct_seconds": f"{jct:.9f}",

                    }

                )



            tool_events_path = output_dir / "tool_events.csv"

            assert tool_events_path.is_file(), tool_events_path



            with tool_events_path.open(

                encoding="utf-8",

                newline="",

            ) as f:

                for r in csv.DictReader(f):

                    v = finite_float(r.get("ttl_seconds"))

                    if v is not None:

                        ttl_seconds_values.append(v)



            kv_path = output_dir / "kv_cache_samples.csv"

            assert kv_path.is_file(), kv_path



            with kv_path.open(

                encoding="utf-8",

                newline="",

            ) as f:

                for r in csv.DictReader(f):

                    for key, value in r.items():

                        if key == "elapsed_seconds":

                            continue

                        v = finite_float(value)

                        if v is not None:

                            kv_values[key].append(v)



            ucm = case_summary.get("ucm") or {}

            for key, value in ucm.items():

                if key in {"what_prune_rate", "what_retain_rate"}:

                    continue

                if isinstance(value, (int, float)) and not isinstance(value, bool):

                    if math.isfinite(float(value)):

                        ucm_sums[key] += float(value)



            ucm_event_aggregates = (

                case_summary.get("ucm_event_aggregates") or {}

            )



            ucm_reason_counts.update(

                ucm_event_aggregates.get("reason_counts") or {}

            )



            events = ucm_event_aggregates.get("events") or {}

            for event_name, event_data in events.items():

                if isinstance(event_data, dict):

                    count = event_data.get("count")

                    if isinstance(count, (int, float)):

                        ucm_event_counts[event_name] += count



            storage = case_summary.get("storage") or {}



            gib = finite_float(storage.get("ssd_store_gib"))

            if gib is not None:

                ssd_gib_values.append(gib)



            byt = finite_float(storage.get("ssd_store_bytes"))

            if byt is not None:

                ssd_bytes_values.append(byt)



        if request_fp is not None:

            request_fp.close()

        jct_fp.close()



        assert total_requests == 14693, (case, total_requests)

        assert total_success == 14693, (case, total_success)

        assert total_failed == 0, (case, total_failed)

        assert len(jct_values) == 1274, (case, len(jct_values))



        prefix_hit_rate = (

            prefix_hit / prefix_query

            if prefix_query > 0

            else None

        )



        ttl_hit_rate = (

            ttl_hit / ttl_matched

            if ttl_matched > 0

            else None

        )



        ttl_timeout_rate = (

            ttl_timeout / ttl_matched

            if ttl_matched > 0

            else None

        )



        ucm_dict = dict(ucm_sums)



        wc = ucm_dict.get("when_candidates", 0.0)

        ws = ucm_dict.get("when_selected", 0.0)

        ucm_dict["when_select_rate"] = (

            ws / wc if wc > 0 else None

        )



        what_candidates = ucm_dict.get("what_candidates", 0.0)

        what_retained = ucm_dict.get("what_retained", 0.0)

        what_pruned = ucm_dict.get("what_pruned", 0.0)



        ucm_dict["what_retain_rate"] = (

            what_retained / what_candidates

            if what_candidates > 0

            else None

        )

        ucm_dict["what_prune_rate"] = (

            what_pruned / what_candidates

            if what_candidates > 0

            else None

        )



        aggregate[case] = {

            "identity": {

                "case": case,

                "name": CASE_NAMES[case],

                "shards": 9,

                "trajectories": 1274,

                "tool_events": 13419,

                "expected_requests": 14693,

                "concurrency": 128,

                "reset_aware_prefix": True,

            },

            "correctness": {

                "requests_total": total_requests,

                "success": total_success,

                "failed": total_failed,

                "success_rate": total_success / total_requests,

            },

            "elapsed_seconds": total_elapsed,

            "tokens": {

                "input": input_tokens,

                "output": output_tokens,

                "total": total_tokens,

                "request_throughput_req_per_s": (

                    total_success / total_elapsed

                ),

                "input_throughput_tok_per_s": (

                    input_tokens / total_elapsed

                ),

                "output_throughput_tok_per_s": (

                    output_tokens / total_elapsed

                ),

                "total_throughput_tok_per_s": (

                    total_tokens / total_elapsed

                ),

            },

            "latency_seconds": {

                "ttft": stats(ttft_values),

                "tpot": stats(tpot_values),

                "e2e": stats(e2e_values),

            },

            "jct_seconds": stats(jct_values),

            "prefix_cache": {

                "query_tokens": prefix_query,

                "hit_tokens": prefix_hit,

                "miss_tokens": prefix_query - prefix_hit,

                "token_hit_rate": prefix_hit_rate,

                "miss_rate": (

                    1.0 - prefix_hit_rate

                    if prefix_hit_rate is not None

                    else None

                ),

            },

            "preemptions": preemptions,

            "ttl": {

                "matched_events": ttl_matched,

                "unmatched_events": ttl_unmatched,

                "hit_count": ttl_hit,

                "timeout_count": ttl_timeout,

                "hit_rate": ttl_hit_rate,

                "timeout_rate": ttl_timeout_rate,

                "seconds": stats(ttl_seconds_values),

                "history_source_counts": dict(ttl_history),

            },

            "kv_sample_metrics": {

                key: stats(values)

                for key, values in sorted(kv_values.items())

            },

            "ucm": ucm_dict,

            "ucm_reason_counts": dict(ucm_reason_counts),

            "ucm_event_counts": dict(ucm_event_counts),

            "storage": {

                "ssd_store_gib_per_shard": stats(ssd_gib_values),

                "ssd_store_bytes_per_shard": stats(ssd_bytes_values),

                "note": (

                    "Per-shard end footprint. Shards ran sequentially "

                    "and backing stores were cleaned between runs; "

                    "do not sum as simultaneous disk requirement."

                ),

            },

        }



    comparisons = {}



    def cmp(name, a, b):

        A = aggregate[a]

        B = aggregate[b]



        comparisons[name] = {

            "new_case": a,

            "baseline_case": b,

            "request_throughput_delta_percent": delta_pct(

                A["tokens"]["request_throughput_req_per_s"],

                B["tokens"]["request_throughput_req_per_s"],

            ),

            "total_token_throughput_delta_percent": delta_pct(

                A["tokens"]["total_throughput_tok_per_s"],

                B["tokens"]["total_throughput_tok_per_s"],

            ),

            "prefix_hit_rate_delta_percentage_points": (

                (

                    A["prefix_cache"]["token_hit_rate"]

                    - B["prefix_cache"]["token_hit_rate"]

                )

                * 100.0

            ),

            "ttft_mean_delta_percent": delta_pct(

                A["latency_seconds"]["ttft"]["mean"],

                B["latency_seconds"]["ttft"]["mean"],

            ),

            "e2e_mean_delta_percent": delta_pct(

                A["latency_seconds"]["e2e"]["mean"],

                B["latency_seconds"]["e2e"]["mean"],

            ),

            "jct_mean_delta_percent": delta_pct(

                A["jct_seconds"]["mean"],

                B["jct_seconds"]["mean"],

            ),

        }



    cmp("C_vs_V", "C", "V")

    cmp("F_vs_V", "F", "V")

    cmp("U_vs_V", "U", "V")

    cmp("F_vs_C", "F", "C")



    final = {

        "benchmark": {

            "name": "Full Runtime-Eligible EnvBench",

            "shards": 9,

            "cases": CASES,

            "trajectories_per_case": 1274,

            "tool_events_per_case": 13419,

            "requests_per_case": 14693,

            "total_requests_all_cases": 58772,

            "concurrency": 128,

            "max_input_tokens": 4000,

            "max_model_len": 4096,

            "reset_aware_prefix": True,

            "aggregation": {

                "throughput": (

                    "sum(successful requests or tokens) / "

                    "sum(workload elapsed seconds)"

                ),

                "latency": (

                    "pooled request-level samples across all 9 shards"

                ),

                "jct": (

                    "reconstructed per shard+job_id from request start "

                    "offset and E2E"

                ),

                "prefix_cache": (

                    "sum(hit tokens) / sum(query tokens)"

                ),

                "ttl": (

                    "counts summed from summaries; TTL seconds pooled "

                    "from tool_events.csv"

                ),

                "kv": (

                    "pooled periodic kv_cache_samples.csv observations"

                ),

            },

        },

        "cases": aggregate,

        "comparisons": comparisons,

    }



    json_path = args.output_dir / "aggregate_summary.json"

    json_path.write_text(

        json.dumps(final, indent=2, ensure_ascii=False) + "\n",

        encoding="utf-8",

    )



    tsv_path = args.output_dir / "aggregate_summary.tsv"

    fields = [

        "case",

        "requests",

        "elapsed_s",

        "req_s",

        "total_tok_s",

        "prefix_hit_rate",

        "ttft_mean_s",

        "ttft_median_s",

        "ttft_p95_s",

        "ttft_p99_s",

        "e2e_mean_s",

        "e2e_median_s",

        "e2e_p95_s",

        "e2e_p99_s",

        "jct_mean_s",

        "jct_median_s",

        "jct_p95_s",

        "jct_p99_s",

        "ttl_hit_rate",

        "kv_mean",

        "kv_p95",

        "kv_p99",

        "kv_max",

        "ucm_when_selected",

        "ssd_gib_mean",

        "ssd_gib_max",

    ]



    with tsv_path.open(

        "w",

        encoding="utf-8",

        newline="",

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=fields,

            delimiter="\t",

        )

        writer.writeheader()



        for case in CASES:

            d = aggregate[case]

            kv = d["kv_sample_metrics"].get(

                "kv_cache_usage_perc",

                {},

            )

            ssd = d["storage"]["ssd_store_gib_per_shard"]



            writer.writerow(

                {

                    "case": case,

                    "requests": d["correctness"]["requests_total"],

                    "elapsed_s": d["elapsed_seconds"],

                    "req_s": d["tokens"]["request_throughput_req_per_s"],

                    "total_tok_s": d["tokens"]["total_throughput_tok_per_s"],

                    "prefix_hit_rate": d["prefix_cache"]["token_hit_rate"],

                    "ttft_mean_s": d["latency_seconds"]["ttft"]["mean"],

                    "ttft_median_s": d["latency_seconds"]["ttft"]["median"],

                    "ttft_p95_s": d["latency_seconds"]["ttft"]["p95"],

                    "ttft_p99_s": d["latency_seconds"]["ttft"]["p99"],

                    "e2e_mean_s": d["latency_seconds"]["e2e"]["mean"],

                    "e2e_median_s": d["latency_seconds"]["e2e"]["median"],

                    "e2e_p95_s": d["latency_seconds"]["e2e"]["p95"],

                    "e2e_p99_s": d["latency_seconds"]["e2e"]["p99"],

                    "jct_mean_s": d["jct_seconds"]["mean"],

                    "jct_median_s": d["jct_seconds"]["median"],

                    "jct_p95_s": d["jct_seconds"]["p95"],

                    "jct_p99_s": d["jct_seconds"]["p99"],

                    "ttl_hit_rate": d["ttl"]["hit_rate"],

                    "kv_mean": kv.get("mean"),

                    "kv_p95": kv.get("p95"),

                    "kv_p99": kv.get("p99"),

                    "kv_max": kv.get("max"),

                    "ucm_when_selected": d["ucm"].get(

                        "when_selected",

                        0,

                    ),

                    "ssd_gib_mean": ssd.get("mean"),

                    "ssd_gib_max": ssd.get("max"),

                }

            )



    md = []
    md.append("# Full EnvBench V/U/C/F 聚合结果")
    md.append("")
    md.append(
        "正式实验包含 1274 个 trajectories、13419 个工具事件和每组 "
        "14693 个请求；完整工作负载划分为 9 个确定性分片，在并发 128 "
        "下启用上下文重置感知回放。"
    )
    md.append("")
    md.append("## 整体性能")
    md.append("")
    md.append(
        "| 配置 | Req/s | Total tok/s | Prefix Cache 命中率 | "
        "Mean TTFT | Mean E2E | Mean JCT |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|")

    for case in CASES:
        d = aggregate[case]
        md.append(
            f"| {case} | "
            f"{fmt(d['tokens']['request_throughput_req_per_s'], 2)} | "
            f"{fmt(d['tokens']['total_throughput_tok_per_s'], 2)} | "
            f"{pct(d['prefix_cache']['token_hit_rate'])} | "
            f"{fmt(d['latency_seconds']['ttft']['mean'])} s | "
            f"{fmt(d['latency_seconds']['e2e']['mean'])} s | "
            f"{fmt(d['jct_seconds']['mean'])} s |"
        )

    md.append("")
    md.append("## 尾延迟")
    md.append("")
    md.append(
        "| 配置 | TTFT P50 | TTFT P95 | TTFT P99 | "
        "E2E P50 | E2E P95 | E2E P99 | "
        "JCT P50 | JCT P95 | JCT P99 |"
    )
    md.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for case in CASES:
        d = aggregate[case]
        t = d["latency_seconds"]["ttft"]
        e = d["latency_seconds"]["e2e"]
        j = d["jct_seconds"]
        md.append(
            f"| {case} | "
            f"{fmt(t['median'])} | {fmt(t['p95'])} | {fmt(t['p99'])} | "
            f"{fmt(e['median'])} | {fmt(e['p95'])} | {fmt(e['p99'])} | "
            f"{fmt(j['median'])} | {fmt(j['p95'])} | {fmt(j['p99'])} |"
        )

    md.append("")
    md.append("## KV Cache、TTL 与 UCM")
    md.append("")
    md.append(
        "| 配置 | KV Mean | KV P95 | KV P99 | KV Max | "
        "TTL 命中率 | UCM 选择 blocks | SSD 平均/分片 | SSD 最大/分片 |"
    )
    md.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for case in CASES:
        d = aggregate[case]
        kv = d["kv_sample_metrics"].get("kv_cache_usage_perc", {})
        ssd = d["storage"]["ssd_store_gib_per_shard"]
        md.append(
            f"| {case} | "
            f"{pct(kv.get('mean'))} | "
            f"{pct(kv.get('p95'))} | "
            f"{pct(kv.get('p99'))} | "
            f"{pct(kv.get('max'))} | "
            f"{pct(d['ttl']['hit_rate'])} | "
            f"{fmt(d['ucm'].get('when_selected', 0), 0)} | "
            f"{fmt(ssd.get('mean'), 2)} GiB | "
            f"{fmt(ssd.get('max'), 2)} GiB |"
        )

    md.append("")
    md.append("## 关键对比")
    md.append("")
    md.append(
        "| 对比 | 请求吞吐 | Mean TTFT | Mean E2E | Mean JCT | "
        "Prefix Cache 命中率 |"
    )
    md.append("|---|---:|---:|---:|---:|---:|")

    for name in ["C_vs_V", "F_vs_V", "U_vs_V", "F_vs_C"]:
        c = comparisons[name]
        label = name.replace("_vs_", " vs ")
        md.append(
            f"| {label} | "
            f"{c['request_throughput_delta_percent']:+.2f}% | "
            f"{c['ttft_mean_delta_percent']:+.2f}% | "
            f"{c['e2e_mean_delta_percent']:+.2f}% | "
            f"{c['jct_mean_delta_percent']:+.2f}% | "
            f"{c['prefix_hit_rate_delta_percentage_points']:+.2f} pp |"
        )

    md.append("")
    md.append("## 聚合说明")
    md.append("")
    md.append(
        "吞吐按所有分片的成功请求或 token 总量除以工作负载回放耗时总和"
        "计算，不包含模型和服务启动时间。延迟分位数从所有请求级样本重新"
        "计算，而不是对分片分位数求平均。"
    )
    md.append("")
    md.append(
        "JCT 在每个分片内按 trajectory 重建，再汇总全部 1274 个 "
        "trajectories。KV Cache 统计来自周期采样。不同分片顺序执行，"
        "并在每个配置结束后清理 SSD backing，因此 SSD 数值按分片独立解释。"
    )
    md.append("")
    md.append(
        "TTL 命中率比较预测 TTL 与回放中经过时间缩放和等待上限处理后的"
        "有效工具时间，不表示原始 EnvBench 工具时长预测准确率。"
    )
    md.append("")
    md.append(
        "本实验使用 EnvBench 派生 trajectory 结构、token 长度、工具顺序"
        "和工具等待时间，并以确定性 token ID 前缀构造请求。结果用于评估"
        "服务系统行为，不评估 EnvBench 语义任务正确率。"
    )

    md_path = args.output_dir / "aggregate_comparison.md"

    md_path.write_text(

        "\n".join(md) + "\n",

        encoding="utf-8",

    )



    print("FULL_ENVBench_AGGREGATE=PASS")

    print("aggregate_json =", json_path)

    print("aggregate_tsv =", tsv_path)

    print("aggregate_md =", md_path)



    for case in CASES:

        d = aggregate[case]

        kv = d["kv_sample_metrics"].get(

            "kv_cache_usage_perc",

            {},

        )

        print(

            case,

            f"req/s={d['tokens']['request_throughput_req_per_s']:.4f}",

            f"prefix_hit={d['prefix_cache']['token_hit_rate']:.6f}",

            f"ttft_mean={d['latency_seconds']['ttft']['mean']:.6f}",

            f"e2e_mean={d['latency_seconds']['e2e']['mean']:.6f}",

            f"jct_mean={d['jct_seconds']['mean']:.6f}",

            f"kv_mean={kv.get('mean')}",

            f"kv_p95={kv.get('p95')}",

            f"ttl_hit={d['ttl']['hit_rate']}",

            f"ucm_selected={d['ucm'].get('when_selected', 0)}",

        )





if __name__ == "__main__":

    main()


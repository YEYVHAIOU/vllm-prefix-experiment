#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, q):
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values):
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    return {
        "count": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": max(xs),
    }


def fmt_num(v, digits=6):
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    try:
        x = float(v)
    except Exception:
        return str(v)
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:.{digits}f}"


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{100.0 * float(v):.2f}%"


def read_env(path):
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def find_latest(root, name):
    xs = list(root.rglob(name))
    return max(xs, key=lambda p: p.stat().st_mtime) if xs else None


def read_csv(path):
    if not path or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def numeric_columns(rows):
    if not rows:
        return {}
    cols = defaultdict(list)
    nonempty = Counter()
    for row in rows:
        for k, raw in row.items():
            if raw is None or str(raw).strip() == "":
                continue
            nonempty[k] += 1
            try:
                cols[k].append(float(raw))
            except Exception:
                pass
    out = {}
    for k, xs in cols.items():
        if nonempty[k] and len(xs) / nonempty[k] >= 0.90:
            out[k] = xs
    return out


def choose_col(rows, exact, contains_any=(), excludes=()):
    if not rows:
        return None
    keys = list(rows[0].keys())
    lower = {k.lower(): k for k in keys}
    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]
    for k in keys:
        lk = k.lower()
        if any(x in lk for x in contains_any) and not any(x in lk for x in excludes):
            return k
    return None


def compute_jct(requests):

    if not requests:

        return None

    required = ("job_id", "request_start_offset_seconds", "e2e_seconds")

    if not all(k in requests[0] for k in required):

        return {

            "available": False,

            "trajectory_column": "job_id" if "job_id" in requests[0] else None,

            "start_column": "request_start_offset_seconds" if "request_start_offset_seconds" in requests[0] else None,

            "end_column": "derived:start+e2e" if "e2e_seconds" in requests[0] else None,

        }

    jobs = defaultdict(lambda: {"starts": [], "ends": []})

    used = 0

    for r in requests:

        try:

            start = float(r["request_start_offset_seconds"])

            e2e = float(r["e2e_seconds"])

        except Exception:

            continue

        end = start + e2e

        if not math.isfinite(start) or not math.isfinite(end) or end < start:

            continue

        jobs[r["job_id"]]["starts"].append(start)

        jobs[r["job_id"]]["ends"].append(end)

        used += 1

    vals = [

        max(v["ends"]) - min(v["starts"])

        for v in jobs.values()

        if v["starts"] and v["ends"]

    ]

    if not vals:

        return {

            "available": False,

            "trajectory_column": "job_id",

            "start_column": "request_start_offset_seconds",

            "end_column": "derived:start+e2e",

        }

    return {

        "available": True,

        "jobs": len(vals),

        "requests_used": used,

        "trajectory_column": "job_id",

        "start_column": "request_start_offset_seconds",

        "end_column": "derived:start+e2e",

        **stats(vals),

    }

def summarize_kv_csv(path):
    rows = read_csv(path)
    nums = numeric_columns(rows)
    keep = {}
    key_terms = ("kv", "block", "running", "waiting", "pinned", "free", "cache", "usage", "occup")
    skip_terms = ("time", "timestamp", "unix", "sample", "index")
    for k, xs in nums.items():
        lk = k.lower()
        if any(t in lk for t in key_terms) and not any(t in lk for t in skip_terms):
            keep[k] = stats(xs)
    return {
        "rows": len(rows),
        "columns": keep,
    }


def parse_prometheus(path):
    out = defaultdict(float)
    if not path or not path.exists():
        return {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if not m:
            continue
        try:
            out[m.group(1)] += float(m.group(2))
        except Exception:
            pass
    return dict(out)


def metrics_delta(before, after):
    b = parse_prometheus(before)
    a = parse_prometheus(after)
    keys = set(a) | set(b)
    d = {}
    wanted = ("ucm", "prefix", "cache", "preempt", "running", "waiting", "kv")
    for k in sorted(keys):
        if any(x in k.lower() for x in wanted):
            d[k] = a.get(k, 0.0) - b.get(k, 0.0)
    return d


def parse_ucm_log(path):
    if not path.exists():
        return {"events": {}, "reason_counts": {}}
    events = defaultdict(lambda: {"count": 0, "sum": defaultdict(float), "max": {}})
    reason_counts = Counter()
    num_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    event_re = re.compile(r"\b(UCM_[A-Z0-9_]+)\b")
    reason_re = re.compile(r"\breason=([^\s]+)")
    for raw in path.read_text(errors="replace").splitlines():
        em = event_re.search(raw)
        if not em:
            continue
        ev = em.group(1)
        rec = events[ev]
        rec['count'] += 1
        for k, sval in num_re.findall(raw):
            try:
                v = float(sval)
            except Exception:
                continue
            rec["sum"][k] += v
            rec["max"][k] = max(v, rec["max"].get(k, v))
        rm = reason_re.search(raw)
        if rm:
            reason_counts[f"{ev}:{rm.group(1)}"] += 1
    normalized = {}
    for ev, rec in sorted(events.items()):
        normalized[ev] = {
            "count": rec['count'],
            "sum": dict(sorted(rec["sum"].items())),
            "max": dict(sorted(rec["max"].items())),
        }
    return {
        "events": normalized,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def ucm_curated(ucm):
    ev = ucm.get("events", {})
    def s(name, key):
        return ev.get(name, {}).get("sum", {}).get(key)
    def c(name):
        return ev.get(name, {}).get("count", 0)
    out = {
        "offload_decisions": c("UCM_OFFLOAD_DECISION"),
        "when_candidates": s("UCM_OFFLOAD_DECISION", "candidates"),
        "when_effective": s("UCM_OFFLOAD_DECISION", "effective"),
        "when_selected": s("UCM_OFFLOAD_DECISION", "selected"),
        "when_skipped": s("UCM_OFFLOAD_DECISION", "skipped"),
        "what_decisions": c("UCM_WHAT_DECISION"),
        "what_candidates": s("UCM_WHAT_DECISION", "candidates"),
        "what_retained": s("UCM_WHAT_DECISION", "retained"),
        "what_pruned": s("UCM_WHAT_DECISION", "pruned"),
        "tier_create_events": c("UCM_TIER_CREATE"),
        "tier_create_total": s("UCM_TIER_CREATE", "total"),
        "tier_create_dram": s("UCM_TIER_CREATE", "dram"),
        "tier_create_ssd": s("UCM_TIER_CREATE", "ssd"),
        "tier_dump_events": c("UCM_TIER_DUMP"),
        "tier_dump_dram_blocks": s("UCM_TIER_DUMP", "dram_blocks"),
        "tier_dump_ssd_blocks": s("UCM_TIER_DUMP", "ssd_blocks"),
        "tier_load_events": c("UCM_TIER_LOAD"),
        "tier_load_dram_blocks": s("UCM_TIER_LOAD", "dram_blocks"),
        "tier_load_ssd_blocks": s("UCM_TIER_LOAD", "ssd_blocks"),
    }
    wc = out["what_candidates"]
    wr = out["what_retained"]
    wp = out["what_pruned"]
    out["what_prune_rate"] = (wp / wc) if wc not in (None, 0) and wp is not None else None
    out["what_retain_rate"] = (wr / wc) if wc not in (None, 0) and wr is not None else None
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize one vLLM/UCM/Continuum benchmark case")
    ap.add_argument("run_dir")
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()
    summary_path = Path(args.summary).resolve() if args.summary else find_latest(run, "summary.json")
    if not summary_path or not summary_path.exists():
        raise SystemExit(f"ERROR: no summary.json under {run}")

    d = json.loads(summary_path.read_text())
    out_dir = summary_path.parent
    requests_csv = out_dir / "requests.csv"
    kv_csv = out_dir / "kv_cache_samples.csv"
    metrics_before = out_dir / "metrics_before.txt"
    metrics_after = out_dir / "metrics_after.txt"
    server_log = run / "server.log"

    declared = read_env(run / "declared_config.env")
    workload = read_env(run / "workload_config.txt")

    req = d.get("requests", {})
    tok = d.get("tokens", {})
    lat = d.get("latency_seconds", {})
    prefix = d.get("prefix_cache", {})
    ttl = d.get("ttl", {})
    exp = d.get("experiment", {})

    query_tokens = prefix.get("query_tokens")
    hit_tokens = prefix.get("hit_tokens")
    miss_tokens = None
    if query_tokens is not None and hit_tokens is not None:
        miss_tokens = query_tokens - hit_tokens

    requests_rows = read_csv(requests_csv)
    jct = compute_jct(requests_rows)
    kv_csv_summary = summarize_kv_csv(kv_csv)
    ucm_raw = parse_ucm_log(server_log)
    ucm = ucm_curated(ucm_raw)
    metric_deltas = metrics_delta(metrics_before, metrics_after)

    ssd_bytes = None
    ssd_path = run / "ssd_store_bytes.txt"
    if ssd_path.exists():
        try:
            ssd_bytes = int(ssd_path.read_text().strip())
        except Exception:
            pass

    runtime_errors = []
    err_file = run / "runtime_errors.txt"
    if err_file.exists():
        runtime_errors = [x for x in err_file.read_text(errors="replace").splitlines() if x.strip()]

    total = req.get("total")
    success = req.get("success")
    failed = req.get("failed")
    success_rate = (success / total) if total else None

    result = {
        "identity": {
            "case": workload.get("case") or declared.get("CASE") or exp.get("experiment_label"),
            "config_name": declared.get("CASE_NAME") or declared.get("LABEL"),
            "label": workload.get("label") or exp.get("experiment_label"),
            "concurrency": exp.get("concurrency"),
            "duration_scale": exp.get("duration_scale"),
            "elapsed_seconds": exp.get("elapsed_seconds"),
            "run_dir": str(run),
            "summary_json": str(summary_path),
        },
        "correctness": {
            "requests_total": total,
            "success": success,
            "failed": failed,
            "success_rate": success_rate,
            "runtime_error_lines": len(runtime_errors),
        },
        "tokens": tok,
        "latency_seconds": lat,
        "prefix_cache": {
            **prefix,
            "miss_tokens": miss_tokens,
            "miss_rate": (miss_tokens / query_tokens) if query_tokens else None,
        },
        "kv_cache_usage_fraction": d.get("kv_cache_usage_fraction"),
        "preemptions": d.get("preemptions"),
        "ttl": ttl,
        "jct_seconds": jct,
        "kv_sample_metrics": kv_csv_summary,
        "ucm": ucm,
        "ucm_event_aggregates": ucm_raw,
        "selected_server_metric_deltas": metric_deltas,
        "storage": {
            "ssd_store_bytes": ssd_bytes,
            "ssd_store_gib": (ssd_bytes / (1024 ** 3)) if ssd_bytes is not None else None,
        },
        "validation": {
            "workload_result": "PASS" if total is not None and success == total and failed == 0 else "FAIL",
            "runtime_errors": "NONE" if not runtime_errors else "FOUND",
            "case_run": "PASS" if total is not None and success == total and failed == 0 and not runtime_errors else "FAIL",
        },
    }

    lines = []
    A = lines.append
    A("=" * 78)
    A("CASE SUMMARY")
    A("=" * 78)
    ident = result["identity"]
    A(f"Case: {ident['case']} ({ident.get('config_name') or 'N/A'})    Label: {ident['label']}    Concurrency: {ident['concurrency']}")
    A(f"Elapsed: {fmt_num(ident['elapsed_seconds'])} s    Duration scale: {ident['duration_scale']}")
    A(f"Run: {run}")
    A("")
    A("[Correctness]")
    A(f"Requests: total={total} success={success} failed={failed} success_rate={fmt_pct(success_rate)}")
    A(f"Runtime error lines: {len(runtime_errors)}")
    A("")
    A("[Tokens / Throughput]")
    A(f"Input tokens:  {tok.get('input')}")
    A(f"Output tokens: {tok.get('output')}")
    A(f"Total tokens:  {tok.get('total')}")
    A(f"Request throughput:      {fmt_num(req.get('throughput_success_req_per_s'))} req/s")
    A(f"Input token throughput:  {fmt_num(tok.get('input_throughput_tok_per_s'))} tok/s")
    A(f"Output token throughput: {fmt_num(tok.get('output_throughput_tok_per_s'))} tok/s")
    A(f"Total token throughput:  {fmt_num(tok.get('total_throughput_tok_per_s'))} tok/s")
    A("")
    A("[Latency seconds]")
    for name in ("ttft", "tpot", "e2e"):
        x = lat.get(name, {}) or {}
        A(
            f"{name.upper():4s}: count={x.get('count')} "
            f"mean={fmt_num(x.get('mean'))} median={fmt_num(x.get('median'))} "
            f"P90={fmt_num(x.get('p90'))} P95={fmt_num(x.get('p95'))} "
            f"P99={fmt_num(x.get('p99'))} max={fmt_num(x.get('max'))}"
        )
    A("")
    A("[Trajectory / JCT seconds]")
    if jct and jct.get("available"):
        A(
            f"jobs={jct.get('jobs')} requests_used={jct.get('requests_used')} "
            f"mean={fmt_num(jct.get('mean'))} median={fmt_num(jct.get('median'))} "
            f"P90={fmt_num(jct.get('p90'))} P95={fmt_num(jct.get('p95'))} "
            f"P99={fmt_num(jct.get('p99'))} max={fmt_num(jct.get('max'))}"
        )
    else:
        A(f"N/A (detected columns: trajectory={jct.get('trajectory_column') if jct else None}, "
          f"start={jct.get('start_column') if jct else None}, end={jct.get('end_column') if jct else None})")
    A("")
    A("[GPU Prefix Cache]")
    A(f"Query tokens: {fmt_num(query_tokens, 0)}")
    A(f"Hit tokens:   {fmt_num(hit_tokens, 0)}")
    A(f"Miss tokens:  {fmt_num(miss_tokens, 0)}")
    A(f"Hit rate:     {fmt_pct(prefix.get('token_hit_rate'))}")
    A(f"Miss rate:    {fmt_pct(result['prefix_cache'].get('miss_rate'))}")
    A("")
    A("[Scheduler / TTL]")
    A(f"Preemptions: {d.get('preemptions')}")
    ttl_sec = ttl.get("seconds", {}) if isinstance(ttl, dict) else {}
    A(f"TTL matched: {ttl.get('matched_events')}/{(ttl.get('matched_events') or 0)+(ttl.get('unmatched_events') or 0)}")
    A(f"TTL hit={ttl.get('hit_count')} timeout={ttl.get('timeout_count')} hit_rate={fmt_pct(ttl.get('hit_rate'))}")
    if ttl_sec:
        A(
            f"TTL seconds: mean={fmt_num(ttl_sec.get('mean'))} median={fmt_num(ttl_sec.get('median'))} "
            f"P90={fmt_num(ttl_sec.get('p90'))} P95={fmt_num(ttl_sec.get('p95'))} "
            f"P99={fmt_num(ttl_sec.get('p99'))} max={fmt_num(ttl_sec.get('max'))}"
        )
    if ttl.get("history_source_counts"):
        A(f"TTL history sources: {json.dumps(ttl.get('history_source_counts'), ensure_ascii=False, sort_keys=True)}")
    A("")
    A("[KV Cache]")
    kv = d.get("kv_cache_usage_fraction") or {}
    if kv:
        A(
            f"Usage fraction: count={kv.get('count')} mean={fmt_num(kv.get('mean'))} "
            f"median={fmt_num(kv.get('median'))} P90={fmt_num(kv.get('p90'))} "
            f"P95={fmt_num(kv.get('p95'))} P99={fmt_num(kv.get('p99'))} max={fmt_num(kv.get('max'))}"
        )
    else:
        A("Usage fraction: N/A")
    for k, st in kv_csv_summary.get("columns", {}).items():
        A(
            f"{k}: mean={fmt_num(st.get('mean'))} median={fmt_num(st.get('median'))} "
            f"P90={fmt_num(st.get('p90'))} P95={fmt_num(st.get('p95'))} "
            f"P99={fmt_num(st.get('p99'))} max={fmt_num(st.get('max'))}"
        )
    A("")
    A("[UCM]")
    if not ucm_raw.get("events"):
        A("No UCM events detected.")
    else:
        A(
            f"WHEN: decisions={ucm.get('offload_decisions')} "
            f"candidate={fmt_num(ucm.get('when_candidates'),0)} "
            f"effective={fmt_num(ucm.get('when_effective'),0)} "
            f"selected={fmt_num(ucm.get('when_selected'),0)} "
            f"skipped={fmt_num(ucm.get('when_skipped'),0)}"
        )
        A(
            f"WHAT: decisions={ucm.get('what_decisions')} "
            f"candidate={fmt_num(ucm.get('what_candidates'),0)} "
            f"retained={fmt_num(ucm.get('what_retained'),0)} "
            f"pruned={fmt_num(ucm.get('what_pruned'),0)} "
            f"prune_rate={fmt_pct(ucm.get('what_prune_rate'))}"
        )
        A(
            f"TIER CREATE: events={ucm.get('tier_create_events')} total={fmt_num(ucm.get('tier_create_total'),0)} "
            f"dram={fmt_num(ucm.get('tier_create_dram'),0)} ssd={fmt_num(ucm.get('tier_create_ssd'),0)}"
        )
        A(
            f"TIER DUMP: events={ucm.get('tier_dump_events')} "
            f"dram_blocks={fmt_num(ucm.get('tier_dump_dram_blocks'),0)} "
            f"ssd_blocks={fmt_num(ucm.get('tier_dump_ssd_blocks'),0)}"
        )
        A(
            f"TIER LOAD: events={ucm.get('tier_load_events')} "
            f"dram_blocks={fmt_num(ucm.get('tier_load_dram_blocks'),0)} "
            f"ssd_blocks={fmt_num(ucm.get('tier_load_ssd_blocks'),0)}"
        )
        if ucm_raw.get("reason_counts"):
            A(f"Reasons: {json.dumps(ucm_raw['reason_counts'], ensure_ascii=False, sort_keys=True)}")
        extra_events = sorted(set(ucm_raw["events"]) - {
            "UCM_OFFLOAD_DECISION", "UCM_WHAT_DECISION", "UCM_TIER_CREATE",
            "UCM_TIER_DUMP", "UCM_TIER_LOAD"
        })
        if extra_events:

            A("Extra UCM events:")

            for x in extra_events:

                rec = ucm_raw["events"][x]

                sums = ", ".join(

                    f"{k}={fmt_num(v, 0)}"

                    for k, v in rec.get("sum", {}).items() if k not in {"pid", "rank", "worker", "device"}

                )

                A(f"  {x}: count={rec['count']}" + (f" sum[{sums}]" if sums else ""))

    A("")
    A("[Storage]")
    A(f"SSD backing bytes: {ssd_bytes if ssd_bytes is not None else 'N/A'}")
    A(f"SSD backing GiB:   {fmt_num(result['storage']['ssd_store_gib'])}")
    A("")
    A("[Validation]")
    A(f"WORKLOAD_RESULT={result['validation']['workload_result']}")
    A(f"RUNTIME_ERRORS={result['validation']['runtime_errors']}")
    A(f"CASE_RUN={result['validation']['case_run']}")
    A("=" * 78)

    text = "\n".join(lines) + "\n"
    (run / "case_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (run / "case_summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

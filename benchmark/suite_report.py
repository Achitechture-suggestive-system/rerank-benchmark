"""Validate and export suite results without rewriting the historical README."""
import csv
import hashlib
import json
from pathlib import Path

from .catalog import catalog_dict
from .metrics import paired_comparison, percentile
from .retrieval import corpus_hash
from .run import sha, write_json


def build_report(root, rows, corpus, dataset):
    from .suite import summarize, validate_records
    root = Path(root)
    reports, paths = [], []
    query_ids = [row["id"] for row in rows]
    lookup = {row["id"]: row for row in rows}
    for stage in ("retrieval", "rerank"):
        for path in sorted((root / stage).rglob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("schema_version") != 2 or report.get("stage") != stage:
                raise ValueError(f"Not a suite artifact: {path}")
            if report["dataset_sha256"] != sha(dataset) or report["corpus_sha256"] != corpus_hash(corpus):
                raise ValueError(f"Dataset/corpus mismatch: {path}")
            if report["query_ids"] != query_ids:
                raise ValueError(f"Query selection mismatch: {path}; smoke and full runs need separate roots")
            validate_records(report, rows, corpus)
            if report["status"] == "complete" and len(report["records"]) != len(rows):
                raise ValueError(f"Complete artifact omits queries: {path}")
            if stage == "retrieval" and report["method"].startswith("rrf-"):
                parents = [root / "retrieval" / (name + ".json") for name in ("bm25", report["method"][4:])]
                current = hashlib.sha256("".join(sha(p) if p.exists() else "missing" for p in parents).encode()).hexdigest()
                if current != report["parent_sha256"]:
                    raise ValueError(f"Fusion parent changed: {path}")
            if stage == "rerank":
                pool = report["retriever"]
                parent_path = root / "retrieval" / (pool + ".json")
                if pool != "oracle":
                    if not parent_path.exists() or sha(parent_path) != report["parent_sha256"]:
                        raise ValueError(f"Retrieval parent changed: {path}")
                    parent_records = {r["id"]: r for r in json.loads(parent_path.read_text(encoding="utf-8"))["records"]}
                for record in report["records"]:
                    expected = ([doc["id"] for doc in lookup[record["id"]]["candidates"]] if pool == "oracle"
                                else parent_records[record["id"]]["ranking"][:report["scoring_config"]["candidate_k"]])
                    if record["candidate_ids"] != expected:
                        raise ValueError(f"Reranker pool differs from retrieval: {path}")
                    if pool != "oracle" and record.get("retrieval_metrics") != parent_records[record["id"]]["metrics"]:
                        raise ValueError(f"Retrieval metrics were changed: {path}")
            reports.append(report)
            paths.append(path)
    if not reports:
        raise ValueError("No suite artifacts; run retrieval or reranking first")
    if len({r["implementation_sha256"] for r in reports}) != 1:
        raise ValueError("Cannot merge artifacts from different implementations; use separate output roots")
    lines = ["# Retrieval + reranking benchmark", "",
             f"Queries: {len(rows)}; corpus: {len(corpus)} unique document IDs.", "",
             "Judgments cover only the original candidate documents per query. Unjudged documents use zero gain in these metrics; Hole@K reports their share. A low score may reflect incomplete judgments. These results do not establish production relevance or calibrated abstention.", "",
             "nDCG uses exponential gain (2^grade - 1), matching this evidence-depth fixture; it is not numerically interchangeable with BEIR/trec_eval linear-gain nDCG. IDCG and recall denominators use all known positives even if retrieval misses them.", "",
             "Quality below uses complete, full-dataset core runs. Long-context and unanswerable diagnostics remain in per-query.csv and JSON. English/Chinese-only models are comparison baselines; tokenizer caps and hardware differ across models.", "",
             "| Run | Status | Completed | Reason |", "| --- | --- | ---: | --- |"]
    for report in reports:
        reason = report.get("reason", "").replace("|", "/").replace("\n", " ")
        state = "partial" if report["status"] == "complete" and not report["full_dataset"] else report["status"]
        lines.append(f"| {report['stage']}/{report['method']} | {state} | {len(report['records'])}/{len(rows)} | {reason} |")
    summaries, comparisons, query_export, breakdown = [], [], [], []
    for title, stage, oracle in (("Retrieval over the full corpus", "retrieval", False),
                                  ("End-to-end retrieval → top-K rerank", "rerank", False),
                                  ("Oracle six-document reranking", "rerank", True)):
        lines.extend(["", "## " + title, "", "| Method | nDCG@10 | MRR@10 | Strict@1 | Recall@20 | Candidate recall | Hole@10 | Δ nDCG@10 | p50 total s/query |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for report in reports:
            if report["stage"] != stage or (stage == "rerank" and (report["retriever"] == "oracle") != oracle):
                continue
            if report["status"] != "complete" or not report["full_dataset"]:
                continue
            core = [r for r in report["records"] if r["track"] == "core"]
            summary = summarize(core)
            comparison = None
            if stage == "rerank" and not oracle:
                base = [dict(id=r["id"], group_id=r["group_id"], track=r["track"], metrics=r["retrieval_metrics"]) for r in core]
                comparison = paired_comparison(base, core, metric="ndcg@10") if core else None
                if comparison:
                    comparisons.append(dict(method=report["method"], candidate_k=report["scoring_config"]["candidate_k"], **comparison))
            delta = f"{comparison['delta']:+.4f}" if comparison else "—"
            p50 = percentile([r.get("online_latency_seconds", r["latency_seconds"]) for r in report["records"]], .5)
            get = lambda key: f"{summary[key]:.4f}" if summary.get(key) is not None else "—"
            lines.append(f"| {report['method']} | {get('ndcg@10')} | {get('mrr@10')} | {get('strict@1')} | {get('recall@20')} | {get('candidate_recall')} | {get('hole@10')} | {delta} | {p50:.3f} |")
            summaries.append(dict(stage=stage, method=report["method"], candidate_k=report["scoring_config"]["candidate_k"],
                                  device=report.get("backend", {}).get("device"), dtype=report.get("backend", {}).get("dtype"),
                                  effective_max_length=report.get("backend", {}).get("max_length"), p50_total_seconds=p50,
                                  delta_ndcg10=comparison["delta"] if comparison else None, **summary))
        if stage == "rerank" and not oracle:
            lines.extend(["", "Δ compares the same queries and retriever before/after reranking. Candidate recall cannot increase when reranking the same top-K pool. Oracle pools and corpus retrieval are different experiments."])
    lines.extend(["", "## Runtime and comparison audit", "",
                  "p50 online total = retrieval + reranking (including API pacing/retry); for RRF it includes the two parent query times plus fusion. These are additive estimates from separate runs, not joint serving measurements. Document indexing, model load and warmup are separate in JSON; API request time includes network, not just server computation. Compare latency only on matched hardware, windows and package versions.", "",
                  "Cluster bootstrap uses scenario groups, keeping paired VI/EN queries together. CI describes this synthetic fixture, not generalization to other datasets.", "",
                  "| Method | Core Δ nDCG@10 | Cluster CI 95% | Improved | Harmed | Unchanged |", "| --- | ---: | --- | ---: | ---: | ---: |"])
    for comparison in comparisons:
        lo, hi = comparison["ci95"]
        lines.append(f"| {comparison['method']} | {comparison['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {comparison['improved']} | {comparison['harmed']} | {comparison['unchanged']} |")
    for report in reports:
        if report["status"] == "complete" and report["full_dataset"]:
            for field in ("language", "category", "track"):
                for value in sorted({r[field] for r in report["records"]}):
                    subset = [r for r in report["records"] if r[field] == value and (field == "track" or r["track"] == "core")]
                    breakdown.append(dict(stage=report["stage"], method=report["method"], dimension=field, value=value, **summarize(subset)))
        for r in report["records"]:
            query_export.append(dict(stage=report["stage"], method=report["method"], status=report["status"],
                                     query_id=r["id"], group_id=r["group_id"], language=r["language"], track=r["track"], category=r["category"],
                                     total_seconds=r["latency_seconds"], retrieval_seconds=r.get("retrieval_latency_seconds"),
                                     online_seconds=r.get("online_latency_seconds", r["latency_seconds"]),
                                     api_request_seconds=r["audit"].get("request_seconds"), rate_limit_wait_seconds=r["audit"].get("rate_limit_wait_seconds"),
                                     attempts=r["audit"].get("attempts"), top1=r["ranking"][0], **r["metrics"]))
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, records in (("summary.csv", summaries), ("per-query.csv", query_export), ("breakdown.csv", breakdown)):
        keys = list(dict.fromkeys(key for record in records for key in record))
        with (root / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    write_json(root / "comparisons.json", comparisons)
    write_json(root / "catalog.json", catalog_dict())
    planned = json.loads((root / "plan.json").read_text(encoding="utf-8")) if (root / "plan.json").exists() else {}
    actual = {p.relative_to(root).as_posix() for p in paths}
    missing = sorted(set(planned.get("expected_artifacts", [])) - actual)
    if missing:
        with (root / "report.md").open("a", encoding="utf-8") as handle:
            handle.write("\n## Planned but missing (not benchmarked)\n\n" + "\n".join("- " + name for name in missing) + "\n")
    write_json(root / "manifest.json", {"dataset_sha256": sha(dataset), "corpus_sha256": corpus_hash(corpus), "missing_planned_artifacts": missing,
                                        "files": {p.relative_to(root).as_posix(): sha(p) for p in paths},
                                        "status_counts": {s: sum(r["status"] == s for r in reports) for s in ("complete", "failed", "skipped", "starting")}})
    print(f"Exported {root / 'report.md'}, summary.csv, per-query.csv, comparisons.json, manifest.json")

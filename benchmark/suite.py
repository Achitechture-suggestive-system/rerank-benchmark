"""Colab suite: full-corpus retrieval, top-K reranking, oracle pools, and reports.

Each model runs in a subprocess. Completed compatible artifacts are reused;
failed prefixes resume only with the same scoring configuration and code hash.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .backends import make_backend
from .catalog import EMBEDDERS, PROFILES, RERANKERS
from .metrics import rank_ids
from .retrieval import DenseIndex, LexicalIndex, corpus_hash, load_collection, retrieval_metrics, rrf_scores
from .run import ROOT, hardware, sha, source_hash, write_json


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def summarize(records):
    keys = sorted({key for record in records for key in record["metrics"]})
    result = {"queries": len(records), "answerable_queries": sum("ndcg@10" in r["metrics"] for r in records)}
    for key in keys:
        values = [r["metrics"][key] for r in records if r["metrics"].get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def scoring_config(args):
    return {key: getattr(args, key) for key in (
        "device", "dtype", "batch_size", "embedding_batch_size", "max_length", "embedding_max_length",
        "threads", "seed", "candidate_k", "rrf_k", "cutoff_layer", "compress_ratio", "compress_layers")}


def prepare(path, stage, method, rows, corpus, args, parent_hash=None):
    current_hardware = hardware()
    expected = dict(stage=stage, method=method, dataset_sha256=sha(args.dataset), corpus_sha256=corpus_hash(corpus),
                    implementation_sha256=source_hash(), query_ids=[r["id"] for r in rows],
                    scoring_config=scoring_config(args), parent_sha256=parent_hash)
    if path.exists():
        if not args.resume:
            raise ValueError(f"Artifact exists: {path}; use --resume or another output root")
        report = json.loads(path.read_text(encoding="utf-8"))
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Incompatible artifact {path}; use a new root for changed code/data/profile")
        records = report["records"]
        if [r["id"] for r in records] != expected["query_ids"][:len(records)]:
            raise ValueError(f"Invalid checkpoint prefix in {path}")
        validate_records(report, rows, corpus)
        if report["status"] == "complete":
            if len(records) != len(rows):
                raise ValueError(f"Incomplete artifact marked complete: {path}")
            return report, True
        if records and report["sessions"][0]["hardware"].get("packages", {}) != current_hardware.get("packages", {}):
            raise ValueError(f"Package versions changed for partial {path}; use original environment or a new root")
        report.setdefault("sessions", []).append({"started_utc": utc(), "resume_from": len(records), "hardware": current_hardware})
        report["status"] = "starting"
        report.pop("reason", None)
    else:
        report = dict(schema_version=2, **expected, status="starting", started_utc=utc(),
                      full_dataset=not bool(args.limit), selected_queries=len(rows), records=[],
                      corpus_documents=len(corpus), judgment_policy="unjudged_as_zero; reported via Hole@K",
                      config=vars(args).copy(), sessions=[{"started_utc": utc(), "resume_from": 0, "hardware": current_hardware}])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    return report, False


def validate_records(report, rows, corpus):
    lookup = {r["id"]: r for r in rows}
    corpus_ids = {doc["id"] for doc in corpus}
    if [r["id"] for r in report["records"]] != report["query_ids"][:len(report["records"])]:
        raise ValueError("Records are not a unique prefix of the query selection")
    for record in report["records"]:
        row = lookup[record["id"]]
        if any(record.get(key) != row[key] for key in ("group_id", "category", "track", "language", "query")):
            raise ValueError("Record metadata disagrees with dataset")
        ids = record["candidate_ids"]
        if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(corpus_ids) or set(record["scores"]) != set(ids):
            raise ValueError("Invalid candidate pool")
        if report["stage"] == "retrieval" and set(ids) != corpus_ids:
            raise ValueError("Retrieval artifact omitted corpus documents")
        if rank_ids(ids, [record["scores"][key] for key in ids]) != record["ranking"]:
            raise ValueError("Ranking disagrees with scores or stable tie order")
        expected = retrieval_metrics(row, record["ranking"], report["scoring_config"]["candidate_k"])
        if expected != record["metrics"]:
            raise ValueError("Stored metrics disagree with judgments")
        if any(not math.isfinite(record[key]) or record[key] < 0 for key in ("latency_seconds",)):
            raise ValueError("Invalid latency")


def record_for(row, ids, scores, elapsed, audit, args):
    ranking = rank_ids(ids, scores)
    return dict(**{key: row[key] for key in ("id", "group_id", "category", "track", "language", "query")},
                scored_utc=utc(),
                candidate_ids=ids, scores=dict(zip(ids, scores)), ranking=ranking,
                latency_seconds=elapsed, audit=audit, metrics=retrieval_metrics(row, ranking, args.candidate_k))


def finalize(path, report, started, status="complete", reason=None):
    report.update(status=status, completed_queries=len(report["records"]), ended_utc=utc())
    report["sessions"][-1]["wall_seconds"] = time.perf_counter() - started
    report["wall_seconds"] = sum(session.get("wall_seconds", 0) for session in report["sessions"])
    if reason:
        report["reason"] = reason
    report["summary"] = summarize(report["records"])
    report["core_summary"] = summarize([r for r in report["records"] if r["track"] == "core"])
    report["by_language_core"] = {lang: summarize([r for r in report["records"] if r["track"] == "core" and r["language"] == lang])
                                  for lang in sorted({r["language"] for r in report["records"]})}
    write_json(path, report)
    print(f"{report['method']}: {status} {report['completed_queries']}/{report['selected_queries']} -> {path}", flush=True)


def retrieval_worker(args, rows, corpus):
    method = args.worker_method
    path = Path(args.output) / "retrieval" / (method + ".json")
    parents = [Path(args.output) / "retrieval" / (name + ".json") for name in ("bm25", method[4:])] if method.startswith("rrf-") else []
    # Fingerprint both parents before reusing a complete fusion checkpoint.
    parent_hash = hashlib.sha256("".join(sha(p) if p.exists() else "missing" for p in parents).encode()).hexdigest() if parents else None
    report, done = prepare(path, "retrieval", method, rows, corpus, args, parent_hash)
    if done:
        print(f"Reusing complete {path}", flush=True)
        return 0
    started = time.perf_counter()
    try:
        if method.startswith("rrf-"):
            payloads = [json.loads(p.read_text(encoding="utf-8")) for p in parents]
            for payload in payloads:
                if any(payload.get(key) != report[key] for key in ("query_ids", "corpus_sha256", "dataset_sha256", "implementation_sha256", "scoring_config")) or payload["status"] != "complete":
                    raise ValueError("RRF requires compatible complete BM25 and dense artifacts")
                validate_records(payload, rows, corpus)
            parent_records = [{r["id"]: r for r in p["records"]} for p in payloads]
            report["backend"] = {"model_id": method, "rrf_constant": args.rrf_k, "parent_hashes": [sha(p) for p in parents]}
            index = None
        else:
            if method in EMBEDDERS and EMBEDDERS[method].vram_gib >= 12 and not args.allow_large_models:
                finalize(path, report, started, "skipped", "Use --allow-large-models on a suitable GPU")
                return 1
            index = LexicalIndex(corpus, method) if method in ("bm25", "tfidf") else DenseIndex(method, corpus, args)
            report["backend"] = index.info
        report["load_and_index_seconds"] = time.perf_counter() - started
        ids = [doc["id"] for doc in corpus]
        for row in rows[len(report["records"]):]:
            before = time.perf_counter()
            if method.startswith("rrf-"):
                ranked = [parent[row["id"]]["ranking"] for parent in parent_records]
                scores = rrf_scores(ids, ranked, args.rrf_k)
                audit = {"rank_only_fusion": True}
            else:
                scores, audit = index.score(row["query"])
            record = record_for(row, ids, scores, time.perf_counter() - before, audit, args)
            if method.startswith("rrf-"):
                record["online_latency_seconds"] = record["latency_seconds"] + sum(parent[row["id"]].get("online_latency_seconds", parent[row["id"]]["latency_seconds"]) for parent in parent_records)
            else:
                record["online_latency_seconds"] = record["latency_seconds"]
            report["records"].append(record)
            write_json(path, report)
            print(f"{method}: {len(report['records'])}/{len(rows)} {row['id']}", flush=True)
        finalize(path, report, started)
        return 0
    except Exception as exc:
        finalize(path, report, started, "failed", f"{type(exc).__name__}: {str(exc)[:1000]}")
        return 1


def rerank_worker(args, rows, corpus):
    model = args.worker_method
    pools = list(args.pools)
    if args.oracle:
        pools.append("oracle")
    texts = {doc["id"]: doc["text"] for doc in corpus}
    jobs, failed = [], False
    for pool in pools:
        source = None if pool == "oracle" else Path(args.output) / "retrieval" / (pool + ".json")
        if source is not None:
            if not source.exists():
                print(f"Unavailable retrieval {pool}: missing artifact", file=sys.stderr)
                failed = True
                continue
            parent = json.loads(source.read_text(encoding="utf-8"))
            if parent["status"] != "complete" or parent["query_ids"] != [r["id"] for r in rows] or parent["dataset_sha256"] != sha(args.dataset) or parent["corpus_sha256"] != corpus_hash(corpus) or parent["implementation_sha256"] != source_hash() or parent["scoring_config"] != scoring_config(args):
                print(f"Unavailable retrieval {pool}: incompatible/incomplete artifact", file=sys.stderr)
                failed = True
                continue
            validate_records(parent, rows, corpus)
            parent_records = {r["id"]: r for r in parent["records"]}
        else:
            parent_records = None
        path = Path(args.output) / "rerank" / model / (pool + ".json")
        report, done = prepare(path, "rerank", model + "/" + pool, rows, corpus, args, sha(source) if source else None)
        if not done:
            report.update(model=model, retriever=pool)
            jobs.append((path, report, parent_records))
    if not jobs:
        return int(failed)
    started = time.perf_counter()
    backend_args = copy.copy(args)
    backend_args.revision = None  # Each model uses its own immutable catalog revision.
    spec = RERANKERS[model]
    try:
        if spec.family == "cohere" and not os.environ.get("COHERE_API_KEY"):
            for path, report, _ in jobs:
                finalize(path, report, started, "skipped", "COHERE_API_KEY is not configured")
            return 1
        if spec.vram_gib >= 12 and not args.allow_large_models:
            for path, report, _ in jobs:
                finalize(path, report, started, "skipped", "Use --allow-large-models on a suitable GPU")
            return 1
        backend = make_backend(model, backend_args)
        load_seconds = time.perf_counter() - started
        # One warm-up per model process, not per retrieval pool.
        warmup_seconds = 0.0
        if spec.family != "cohere" or not any(report["records"] for _, report, _ in jobs):
            warm = time.perf_counter()
            backend.score("Which city is the capital of France?", ["Paris is the capital of France.", "A banana is a fruit."])
            warmup_seconds = time.perf_counter() - warm
    except Exception as exc:
        for path, report, _ in jobs:
            finalize(path, report, started, "failed", f"{type(exc).__name__}: {str(exc)[:1000]}")
        return 1
    for path, report, parent_records in jobs:
        job_started = time.perf_counter()
        report.update(backend=backend.info, load_seconds=load_seconds, warmup_seconds=warmup_seconds)
        try:
            for row in rows[len(report["records"]):]:
                ids = ([doc["id"] for doc in row["candidates"]] if parent_records is None
                       else parent_records[row["id"]]["ranking"][:args.candidate_k])
                before = time.perf_counter()
                scores, audit = backend.score(row["query"], [texts[key] for key in ids])
                record = record_for(row, ids, scores, time.perf_counter() - before, audit, args)
                if parent_records is not None:
                    record["retrieval_metrics"] = parent_records[row["id"]]["metrics"]
                    record["retrieval_latency_seconds"] = parent_records[row["id"]].get("online_latency_seconds", parent_records[row["id"]]["latency_seconds"])
                record["online_latency_seconds"] = record["latency_seconds"] + record.get("retrieval_latency_seconds", 0)
                report["records"].append(record)
                write_json(path, report)
                print(f"{report['method']}: {len(report['records'])}/{len(rows)} {row['id']}", flush=True)
            finalize(path, report, job_started)
        except Exception as exc:
            finalize(path, report, job_started, "failed", f"{type(exc).__name__}: {str(exc)[:1000]}")
            failed = True
    return int(failed)


def selection(args):
    embed = args.embedders if args.embedders is not None else PROFILES[args.profile]["embedders"]
    rerank = args.models if args.models is not None else PROFILES[args.profile]["rerankers"]
    if embed == ["all"]:
        embed = list(EMBEDDERS)
    if rerank == ["all"]:
        rerank = list(RERANKERS)
    for value in embed:
        if value not in EMBEDDERS:
            raise ValueError(f"Unknown embedder: {value}")
    for value in rerank:
        if value not in RERANKERS:
            raise ValueError(f"Unknown reranker: {value}")
    return embed, rerank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "retrieve", "rerank", "report"])
    parser.add_argument("--profile", choices=PROFILES, default="t4")
    parser.add_argument("--dataset", default=str(ROOT / "data/evidence_depth.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedders", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--pools", nargs="*", default=["bm25", "rrf-bge-m3-dense"], help="Retrieval pools for reranking; 'all' uses all compatible retrieval artifacts")
    parser.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-large-models", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--ephemeral-cache", help="Use and remove a task-created temporary HF cache per worker under this directory")
    parser.add_argument("--cohere-min-interval", type=float, default=6.2)
    parser.add_argument("--cohere-max-attempts", type=int, default=8)
    parser.add_argument("--cutoff-layer", type=int, default=28)
    parser.add_argument("--compress-ratio", type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument("--compress-layers", type=int, nargs="+", default=[24])
    parser.add_argument("--worker-method", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.candidate_k, args.embedding_batch_size, args.batch_size, args.threads, args.rrf_k, args.cohere_max_attempts) < 1 or args.limit < 0 or args.max_length < 128 or args.embedding_max_length < 32 or not math.isfinite(args.cohere_min_interval) or args.cohere_min_interval < 0:
        parser.error("Invalid resource, window, query, or pacing argument")
    if not 8 <= args.cutoff_layer <= 42 or any(n not in [8, 16, 24, 32, 40] for n in args.compress_layers):
        parser.error("Invalid Gemma layer configuration")
    rows, corpus = load_collection(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    if args.stage == "report":
        from .suite_report import build_report
        build_report(Path(args.output), rows, corpus, args.dataset)
        return 0
    embed, rerank = selection(args)
    if args.pools == ["all"]:
        args.pools = [p.stem for p in sorted((Path(args.output) / "retrieval").glob("*.json"))
                      if json.loads(p.read_text(encoding="utf-8")).get("status") == "complete"]
    if args.stage == "plan":
        print(f"Dataset: {len(rows)} queries; corpus: {len(corpus)} documents; rerank top-{args.candidate_k}")
        print("Retrieval:", ", ".join(["bm25", "tfidf"] + embed + ["rrf-" + name for name in embed]))
        print("Rerank:", ", ".join(rerank))
        print("Pools:", ", ".join(args.pools + (["oracle"] if args.oracle else [])))
        for name in rerank:
            spec = RERANKERS[name]
            print(f"{name}: env={spec.environment}, VRAM hint={spec.vram_gib:g} GiB, source={spec.source}")
        hosted = sum(RERANKERS[name].family == "cohere" for name in rerank)
        print("Hosted request estimate (before retries):", hosted * (len(rows) * (len(args.pools) + int(args.oracle)) + 1))
        return 0
    if args.worker_method:
        return retrieval_worker(args, rows, corpus) if args.stage == "retrieve" else rerank_worker(args, rows, corpus)
    methods = ["bm25", "tfidf"] + embed + ["rrf-" + name for name in embed] if args.stage == "retrieve" else rerank
    failed = False
    for name in methods:
        # Send a single worker method, preserving all other options and environment.
        command = [sys.executable, "-m", "benchmark.suite", *sys.argv[1:], "--worker-method", name]
        if args.pools != ["all"] and "--pools" not in sys.argv:
            command.extend(["--pools", *args.pools])
        if args.ephemeral_cache and name not in ("bm25", "tfidf") and not name.startswith(("rrf-", "cohere-")):
            Path(args.ephemeral_cache).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="benchmark-model-", dir=args.ephemeral_cache) as cache:
                env = os.environ.copy()
                env.update(HF_HOME=cache, HF_HUB_CACHE=str(Path(cache) / "hub"), HF_MODULES_CACHE=str(Path(cache) / "modules"))
                result = subprocess.run(command, env=env)
        else:
            result = subprocess.run(command)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

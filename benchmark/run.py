"""Run each backend in a separate process to release model memory reliably."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .backends import MODEL_IDS, make_backend
from .metrics import aggregate, percentile, query_metrics, rank_ids, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "benchmark").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def hardware():
    info = {"platform": platform.platform(), "python": sys.version, "cpu": platform.processor(), "logical_cpus": os.cpu_count()}
    info["packages"] = {}
    for package in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        info["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "unavailable"
    if platform.system() == "Windows":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(key, ctypes.c_ulonglong) for key in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            info.update(ram_total_bytes=status.total_phys, ram_available_bytes=status.avail_phys)
    return info


def worker(args):
    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_dataset(rows)
    if args.limit:
        rows = rows[:args.limit]
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    name = args.models[0]
    target = outdir / (name + ".json")
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}; choose a new --output directory")
    report = dict(schema_version=1, model=name, model_id=MODEL_IDS[name], status="starting",
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  dataset_sha256=sha(dataset), implementation_sha256=source_hash(),
                  query_ids=[r["id"] for r in rows], selected_queries=len(rows), full_dataset=not bool(args.limit),
                  hardware=hardware(), config=vars(args).copy(), records=[])
    write_json(target, report)
    start = time.perf_counter()
    backend = None
    try:
        if name in ("qwen4b", "gemma") and not args.allow_large_models:
            report.update(status="skipped", reason="Large model loading disabled for this resource-constrained run; rerun on a suitable machine with --allow-large-models")
            return
        if name.startswith("cohere-") and not os.environ.get("COHERE_API_KEY"):
            report.update(status="skipped", reason="COHERE_API_KEY is not configured")
            return
        load_start = time.perf_counter()
        backend = make_backend(name, args)
        report["load_seconds"] = time.perf_counter() - load_start
        report["backend"] = backend.info
        warm = time.perf_counter()
        backend.score("Which city is the capital of France?", ["Paris is the capital of France.", "A banana is a fruit."])
        report["warmup_seconds"] = time.perf_counter() - warm
        if args.device == "cuda" and hasattr(backend, "torch"):
            backend.torch.cuda.reset_peak_memory_stats()
        for index, row in enumerate(rows):
            # Deliberately strip judgments/rationales/IDs at this boundary.
            before = time.perf_counter()
            values, audit = backend.score(row["query"], [d["text"] for d in row["candidates"]])
            elapsed = time.perf_counter() - before
            ids = [d["id"] for d in row["candidates"]]
            ranking = rank_ids(ids, values)
            scores = dict(zip(ids, values))
            record = {k: row[k] for k in ("id", "group_id", "category", "language", "track", "query")}
            record.update(candidate_ids=ids, ranking=ranking, scores=scores, latency_seconds=elapsed,
                          audit=audit, metrics=query_metrics(row, ranking, scores))
            report["records"].append(record)
            print(f"{name}: {index + 1}/{len(rows)} {row['id']} {elapsed:.3f}s", flush=True)
            write_json(target, report)
        report["summary"] = aggregate(report["records"])
        report["by_track"] = {track: aggregate([r for r in report["records"] if r["track"] == track]) for track in sorted({r["track"] for r in rows})}
        report["by_category"] = {category: aggregate([r for r in report["records"] if r["category"] == category]) for category in sorted({r["category"] for r in rows})}
        report["by_language_core"] = {lang: aggregate([r for r in report["records"] if r["language"] == lang and r["track"] == "core"]) for lang in ("vi", "en")}
        timings = [r["latency_seconds"] for r in report["records"]]
        report["performance"] = dict(p50_query_seconds=percentile(timings, .5), p95_query_seconds=percentile(timings, .95),
                                     scored_pairs=sum(len(r["scores"]) for r in report["records"]), scoring_seconds=sum(timings),
                                     pairs_per_second=sum(len(r["scores"]) for r in report["records"]) / sum(timings),
                                     timing_repetitions=1, includes_tokenization=True, includes_network=name.startswith("cohere-"))
        report["status"] = "complete"
        if args.device == "cuda" and hasattr(backend, "torch"):
            report["performance"]["peak_torch_allocated_bytes"] = backend.torch.cuda.max_memory_allocated()
    except Exception as exc:
        # Do not leak credential-bearing HTTP request headers or environment dumps.
        report.update(status="failed", reason=f"{type(exc).__name__}: {str(exc)[:1500]}")
        print(report["reason"], file=sys.stderr, flush=True)
    finally:
        report["completed_queries"] = len(report["records"])
        report["wall_seconds"] = time.perf_counter() - start
        report["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(target, report)
        print(f"{name}: {report['status']} -> {target}", flush=True)
    return 1 if report["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=["bm25", "bge-m3", "qwen4b", "gemma", "cohere-pro"])
    parser.add_argument("--dataset", default=str(ROOT / "data" / "evidence_depth.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only; reports identify partial datasets")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-large-models", action="store_true")
    parser.add_argument("--revision", help="Hugging Face revision; Gemma requires an immutable SHA")
    parser.add_argument("--cutoff-layer", type=int, default=28)
    parser.add_argument("--compress-ratio", type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument("--compress-layers", type=int, nargs="+", default=[24])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 128 or args.threads < 1 or args.limit < 0:
        parser.error("Invalid batch size, context length, thread count, or query limit")
    if not 8 <= args.cutoff_layer <= 42 or any(n not in [8, 16, 24, 32, 40] for n in args.compress_layers):
        parser.error("Invalid Gemma layer configuration")
    if args.worker:
        if len(args.models) != 1:
            parser.error("Worker requires exactly one model")
        return worker(args)
    argv = sys.argv[1:]
    # Remove the multi-value --models argument; child sees exactly one model.
    if "--models" in argv:
        start = argv.index("--models")
        end = start + 1
        while end < len(argv) and not argv[end].startswith("--"):
            end += 1
        argv = argv[:start] + argv[end:]
    failed = False
    for name in args.models:
        completed = subprocess.run([sys.executable, "-m", "benchmark.run", *argv, "--models", name, "--worker"])
        failed |= completed.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

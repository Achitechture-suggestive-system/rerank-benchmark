"""Print all evidence and scores for a single completed query."""
import argparse
import json
import sys
from pathlib import Path

from .run import sha


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--dataset", default="data/evidence_depth.jsonl")
    args = parser.parse_args()
    report = json.loads(Path(args.result).read_text(encoding="utf-8"))
    if sha(args.dataset) != report["dataset_sha256"]:
        raise ValueError("Dataset does not match this result")
    records = {r["id"]: r for r in report["records"]}
    if args.query not in records:
        raise SystemExit("No completed record for that query")
    result = records[args.query]
    dataset = [json.loads(s) for s in Path(args.dataset).read_text(encoding="utf-8").splitlines()]
    row = next(r for r in dataset if r["id"] == args.query)
    docs = {d["id"]: d for d in row["candidates"]}
    print(row["query"])
    print(f"Model: {report['model_id']} | score type: {report['backend']['score_kind']}")
    print("Metrics:", json.dumps(result["metrics"], ensure_ascii=False))
    for rank, key in enumerate(result["ranking"], 1):
        doc = docs[key]
        print(f"\n#{rank} {key} | score={result['scores'][key]:.6f} | grade={doc['grade']}")
        print(doc["text"])
        print("Rationale:", doc["rationale"])
    print("\nToken audit:", json.dumps(result["audit"], ensure_ascii=False))


if __name__ == "__main__":
    main()

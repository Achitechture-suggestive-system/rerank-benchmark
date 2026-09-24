"""Generate README tables from complete artifacts; reject mismatched datasets."""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from .metrics import aggregate, paired_comparison, query_metrics, rank_ids
from .run import sha

BEGIN, END = "<!-- BENCHMARK_RESULTS_START -->", "<!-- BENCHMARK_RESULTS_END -->"
DISPLAY_NAMES = {
    "bge-m3": "BGE M3",
    "bm25": "BM25",
    "qwen4b": "Qwen3 4B",
    "gemma": "BGE Gemma2 lightweight",
    "cohere-pro": "Cohere 4.0 pro",
    "cohere-fast": "Cohere 4.0 fast",
}


def read_reports(paths, dataset):
    reports = []
    seen = set()
    rows = {r["id"]: r for r in dataset}
    for directory in paths:
        for path in sorted(Path(directory).glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if "schema_version" not in report or "model" not in report:
                continue
            if report["model"] in seen:
                raise ValueError("Duplicate model artifacts; select exactly one run per model")
            seen.add(report["model"])
            report["artifact"] = path.as_posix()
            if report["status"] == "complete":
                records = report["records"]
                if len({r["id"] for r in records}) != len(records) or {r["id"] for r in records} != set(report["query_ids"]):
                    raise ValueError("Missing or duplicate completed records")
                if report["full_dataset"] and set(report["query_ids"]) != set(rows):
                    raise ValueError("Full run omits dataset queries")
                for record in records:
                    ids = [d["id"] for d in rows[record["id"]]["candidates"]]
                    if rank_ids(ids, [record["scores"][key] for key in ids]) != record["ranking"]:
                        raise ValueError("Stored ranking disagrees with scores or tie order")
                    expected = query_metrics(rows[record["id"]], record["ranking"], record["scores"])
                    if expected != record["metrics"]:
                        raise ValueError("Stored metrics disagree with the scored ranking")
            reports.append(report)
    return reports


def generate(reports, dataset):
    lookup = {r["id"]: r for r in dataset}
    complete = [r for r in reports if r["status"] == "complete" and r["full_dataset"]]
    hashes = {r["dataset_sha256"] for r in reports}
    if len(hashes) != 1:
        raise ValueError("Cannot compare different dataset hashes")
    caps = {r["config"]["max_length"] for r in complete if r["model"] != "bm25"}
    if len(caps) > 1:
        raise ValueError("Different token-budget profiles; generate separate reports")
    core_rows = [r for r in dataset if r["track"] == "core"]
    group_count = len({r["group_id"] for r in core_rows})
    lines = ["### 4.1. Chất lượng và trạng thái chạy", "",
             f"Bảng chất lượng dùng **{len(core_rows)} câu core / {group_count} nhóm độc lập**. Hai ngôn ngữ của cùng tình huống không phải hai mẫu độc lập.", "",
             "| Mô hình | Trạng thái |", "| --- | --- |"]
    for report in reports:
        name = DISPLAY_NAMES.get(report["model"], report["model_id"])
        state = report["status"] if report["status"] != "complete" or report["full_dataset"] else "partial dataset"
        lines.append(f"| {name} | `{state}` |")
    lines += ["", "**Điểm chất lượng — chỉ các lần chạy đầy đủ**", "",
              "| Mô hình | nDCG@5 | MRR@5 | Hit@1 | Strict@1 | Hard-neg win |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        if report not in complete:
            continue
        name = DISPLAY_NAMES.get(report["model"], report["model_id"])
        core = aggregate([r for r in report["records"] if r["track"] == "core"])
        lines.append(f"| {name} | {core['ndcg5']:.4f} | {core['mrr5']:.4f} | {core['hit1']:.2%} | {core['strict1']:.2%} | {core['hard_negative_win']:.2%} |")
    lines += ["", "> **Cách đọc:** Hit@1 nhận grade ≥ 2; Strict@1 chỉ nhận grade = 3. Kết quả được sinh từ JSON; mô hình chưa chạy không được gán điểm.", ""]
    for report in reports:
        if report not in complete:
            lines.append(f"- `{report['model']}`: {report.get('reason', 'Partial dataset; excluded from headline')}.")
    lines += ["", "### 4.2. Thời gian xử lý", "",
              "| Mô hình | p50 (s/query) | p95 (s/query) | Cặp/giây |",
              "| --- | ---: | ---: | ---: |"]
    for report in complete:
        name = DISPLAY_NAMES.get(report["model"], report["model_id"])
        perf = report["performance"]
        lines.append(f"| {name} | {perf['p50_query_seconds']:.3f} | {perf['p95_query_seconds']:.3f} | {perf['pairs_per_second']:.2f} |")
    lines += ["", "p50/p95 tính trên toàn bộ query, một lượt sau warm-up; có tokenization, không có thời gian load. Cohere nếu chạy gồm cả network. Đây không phải phép đo throughput production hay so sánh latency trên cùng phần cứng.", "",
              "### 4.3. Phân tích từng mô hình", ""]
    for report in complete:
        records, name = report["records"], report["model"]
        core = [r for r in records if r["track"] == "core"]
        by_category = defaultdict(list)
        for record in core:
            by_category[record["category"]].append(record)
        lines += [f"#### {DISPLAY_NAMES.get(name, name)}", "", "**Theo nhóm năng lực**", "",
                  "| Nhóm | Số query | nDCG@5 | Strict@1 |", "| --- | ---: | ---: | ---: |"]
        for category, values in sorted(by_category.items()):
            result = aggregate(values)
            lines.append(f"| {category} | {result['queries']} | {result['ndcg5']:.4f} | {result['strict1']:.2%} |")
        langs = [(lang, aggregate([r for r in core if r["language"] == lang])) for lang in ("vi", "en")]
        lines += ["", "**Theo ngôn ngữ — chỉ core**", "",
                  "| Ngôn ngữ | nDCG@5 | Strict@1 |", "| --- | ---: | ---: |"]
        lines += [f"| `{lang}` | {m['ndcg5']:.4f} | {m['strict1']:.2%} |" for lang, m in langs]
        lines += ["", "Không diễn giải đây là đánh giá tổng quát mọi dữ liệu tiếng Việt/Anh.", ""]
        known = [r["audit"]["truncated_pairs"] for r in records if r["audit"]["truncated_pairs"] is not None]
        core_known = [r["audit"]["truncated_pairs"] for r in core if r["audit"]["truncated_pairs"] is not None]
        truncated = str(sum(known)) if len(known) == len(records) else "không quan sát được đầy đủ"
        core_truncated = str(sum(core_known)) if len(core_known) == len(core) else "không quan sát được đầy đủ"
        long = aggregate([r for r in records if r["track"] == "long_context"])
        lines += ["**Tài liệu dài và token bị cắt**", "",
                  f"- Long-context: nDCG@5 **{long['ndcg5']:.4f}**, Strict@1 **{long['strict1']:.2%}** trên {long['queries']} câu.",
                  f"- Số cặp bị cắt: toàn bộ **{truncated}**, core **{core_truncated}**.", "",
                  "Nếu bằng chứng bị cắt mất, lỗi thuộc cả chính sách cửa sổ và mô hình; không kết luận mô hình không hiểu bằng chứng chưa được thấy.", ""]
        wrong = sorted([r for r in core if r["metrics"]["strict1"] == 0], key=lambda r: (r["metrics"]["ndcg5"], r["id"]))[:4]
        lines += ["**Một số ca sai có thể kiểm tra lại**", ""] if wrong else ["Không có lỗi Strict@1 trong core của lần chạy này.", ""]
        for record in wrong:
            row = lookup[record["id"]]
            gold = next(d for d in row["candidates"] if d["grade"] == 3)
            top = next(d for d in row["candidates"] if d["id"] == record["ranking"][0])
            gold_rank = record["ranking"].index(gold["id"]) + 1
            lines += [f"- **`{record['id']}`** — top 1 grade **{top['grade']}**, đáp án đầy đủ ở hạng **{gold_rank}**.",
                      f"  {top['rationale']} Điểm top 1: `{record['scores'][top['id']]:.6f}`; điểm đáp án: `{record['scores'][gold['id']]:.6f}`.", ""]
        ood = [r for r in records if r["track"] == "unanswerable"]
        if ood:
            lines += ["**Câu hỏi không có đáp án**", "",
                      f"Unanswerable: **{len(ood)}** câu không có tài liệu đúng. Giá trị lớn nhất trong các điểm top 1 là `{max(r['scores'][r['ranking'][0]] for r in ood):.6f}`.", "",
                      "Chỉ là chẩn đoán trong thang điểm của mô hình; chưa có ngưỡng abstain được hiệu chuẩn và không đưa các câu này vào nDCG/MRR.", ""]
        perf, back = report["performance"], report["backend"]
        lines += ["**Runtime và artifact**", "",
                  f"- Thiết bị: `{back.get('device')}`; dtype: `{back.get('dtype', 'n/a')}`.",
                  f"- Load: {report['load_seconds']:.2f}s; warm-up: {report['warmup_seconds']:.2f}s.",
                  f"- Inference: {perf['scored_pairs']} cặp / {perf['scoring_seconds']:.2f}s = {perf['pairs_per_second']:.2f} cặp/s.",
                  f"- Artifact: [{name}.json]({report['artifact']}).", ""]
    named = {r["model"]: r for r in complete}
    pairs = []
    if "bm25" in named and "bge-m3" in named:
        pairs.append(("bm25", "bge-m3"))
    if "bge-m3" in named:
        pairs += [("bge-m3", name) for name in named if name not in ("bm25", "bge-m3")]
    lines += ["### 4.4. So sánh theo từng query", ""]
    for base, name in pairs:
        comparison = paired_comparison(named[base]["records"], named[name]["records"])
        lo, hi = comparison["ci95"]
        lines += [f"#### {DISPLAY_NAMES.get(name, name)} so với {DISPLAY_NAMES.get(base, base)}", "",
                  f"- Δ nDCG@5 core: **{comparison['delta']:+.4f}**.",
                  f"- Bootstrap CI 95%: **[{lo:+.4f}, {hi:+.4f}]**.",
                  f"- Theo query: **{comparison['improved']} tăng**, **{comparison['harmed']} giảm**, **{comparison['unchanged']} không đổi**.", "",
                  f"Bootstrap {comparison['samples']} lần theo {comparison['groups']} nhóm tình huống, seed cố định. Khoảng này chỉ mô tả biến thiên trong bộ fixture tổng hợp, không suy rộng ra production.", ""]
        a = {r["id"]: r for r in named[base]["records"] if r["track"] == "core"}
        b = {r["id"]: r for r in named[name]["records"] if r["track"] == "core"}
        changes = sorted((b[k]["metrics"]["ndcg5"] - a[k]["metrics"]["ndcg5"], k) for k in a)
        lines += ["| Query minh họa | Δ nDCG@5 |", "| --- | ---: |"]
        selected = [item for item in changes if item[0] < -1e-10][:3] + [item for item in reversed(changes) if item[0] > 1e-10][:3]
        lines += [f"| `{key}` | {delta:+.4f} |" for delta, key in selected]
        lines.append("")
    lines += ["### 4.5. Giới hạn kết luận của lần chạy", "",
              "Chỉ so sánh chất lượng các mô hình có trạng thái complete. Chưa có đủ kết quả Qwen 4B, Gemma2 lightweight và Cohere thì chưa thể xếp hạng bốn mô hình. BM25 là mốc từ vựng, không thay thế một trong bốn mô hình được yêu cầu. Nhãn do tác giả fixture xây dựng và chưa qua đánh giá mù bởi nhiều người.", "",
              "**Dataset SHA-256**", "", "```text", next(iter(hashes)), "```", "",
              "Mỗi JSON lưu revision mô hình, phiên bản thư viện, cấu hình, thời điểm, runtime, score/rank và token audit."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--dataset", default="data/evidence_depth.jsonl")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--output", default="results/report.md")
    args = parser.parse_args()
    dataset = [json.loads(s) for s in Path(args.dataset).read_text(encoding="utf-8").splitlines()]
    reports = read_reports(args.results, dataset)
    if not reports or any(r["dataset_sha256"] != sha(args.dataset) for r in reports):
        raise ValueError("Missing reports or dataset hash mismatch")
    content = generate(reports, dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def with_relative_links(markdown, destination):
        for report in reports:
            relative = Path(os.path.relpath(report["artifact"], Path(destination).resolve().parent)).as_posix()
            markdown = markdown.replace("](" + report["artifact"] + ")", "](" + relative + ")")
        return markdown
    output.write_text(with_relative_links(content, output) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "query", "track", "language", "ndcg5", "mrr5", "strict1", "latency_s", "top1", "top1_score"])
        for report in reports:
            for record in report["records"]:
                metrics = record["metrics"] or {}
                top = record["ranking"][0]
                writer.writerow([report["model"], record["id"], record["track"], record["language"], metrics.get("ndcg5"), metrics.get("mrr5"), metrics.get("strict1"), record["latency_seconds"], top, record["scores"][top]])
    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) >= text.index(END):
        raise ValueError("README must contain exactly one ordered results marker pair")
    before, rest = text.split(BEGIN)
    _, after = rest.split(END)
    readme.write_text(before + BEGIN + "\n\n" + with_relative_links(content, readme) + "\n\n" + END + after, encoding="utf-8")
    print(f"Updated {readme}; {output}; {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

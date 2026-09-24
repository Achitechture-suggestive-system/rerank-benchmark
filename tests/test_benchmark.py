import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark.backends import BM25, cohere_scores
from benchmark.metrics import aggregate, paired_comparison, query_metrics, rank_ids, validate_dataset
from benchmark.report import read_reports, generate
from benchmark.run import ROOT


def fixture():
    return dict(id="x", group_id="g", query="q", category="test", track="core", language="en",
                candidates=[dict(id=str(i), text=f"doc {i}", grade=g, rationale="label") for i, g in enumerate([3, 2, 0, 0])])


class MetricsTests(unittest.TestCase):
    def test_perfect_and_reversed_rankings(self):
        row = fixture()
        scores = {"0": 4, "1": 3, "2": 2, "3": 1}
        result = query_metrics(row, ["0", "1", "2", "3"], scores)
        self.assertEqual(result["ndcg5"], 1)
        self.assertEqual(result["mrr5"], 1)
        self.assertEqual(result["hard_negative_win"], 1)
        reversed_scores = {k: -v for k, v in scores.items()}
        result = query_metrics(row, ["3", "2", "1", "0"], reversed_scores)
        expected = (3 / math.log2(4) + 7 / math.log2(5)) / (7 + 3 / math.log2(3))
        self.assertAlmostEqual(result["ndcg5"], expected)
        self.assertAlmostEqual(result["mrr5"], 1 / 3)
        self.assertEqual(result["strict1"], 0)
        self.assertEqual(result["hard_negative_win"], 0)

    def test_ties_nan_and_missing_results(self):
        self.assertEqual(rank_ids(["b", "a"], [1, 1]), ["b", "a"])
        with self.assertRaises(ValueError):
            rank_ids(["a"], [float("nan")])
        with self.assertRaises(ValueError):
            rank_ids(["a", "b"], [1])
        row = fixture()
        result = query_metrics(row, ["0", "1", "2", "3"], dict.fromkeys(["0", "1", "2", "3"], 1))
        self.assertEqual(result["hard_negative_win"], .5)

    def test_unanswerable_excluded(self):
        row = fixture()
        row["track"] = "unanswerable"
        for d in row["candidates"]:
            d["grade"] = 0
        validate_dataset([row])
        self.assertIsNone(query_metrics(row, ["0", "1", "2", "3"], dict.fromkeys(["0", "1", "2", "3"], 99)))
        self.assertEqual(aggregate([{"metrics": None}]), {"queries": 0})

    def test_cluster_bootstrap_and_query_mismatch(self):
        a = [dict(id=str(i), group_id=str(i // 2), track="core", metrics={"ndcg5": .25}) for i in range(6)]
        b = [{**r, "metrics": {"ndcg5": .75}} for r in a]
        result = paired_comparison(a, b)
        self.assertEqual(result["delta"], .5)
        self.assertEqual(result["ci95"], [.5, .5])
        self.assertEqual(result["groups"], 3)
        with self.assertRaises(ValueError):
            paired_comparison(a, b[:-1])


class ContractTests(unittest.TestCase):
    def test_provider_indices_are_mapped_back(self):
        result = {"results": [{"index": 2, "relevance_score": .9}, {"index": 0, "relevance_score": .4}, {"index": 1, "relevance_score": .1}]}
        self.assertEqual(cohere_scores(result, 3), [.4, .1, .9])
        with self.assertRaises(ValueError):
            cohere_scores(result, 4)

    def test_bm25_reordering_and_relevant_document(self):
        model = BM25()
        scores, _ = model.score("red apple", ["red apple", "blue boat", "apple tree"])
        reversed_scores, _ = model.score("red apple", ["apple tree", "blue boat", "red apple"])
        self.assertEqual(scores, list(reversed(reversed_scores)))
        self.assertGreater(scores[0], scores[1])

    def test_real_dataset_integrity(self):
        rows = [json.loads(line) for line in (ROOT / "data/evidence_depth.jsonl").read_text(encoding="utf-8").splitlines()]
        validate_dataset(rows)
        self.assertEqual(len(rows), 56)
        self.assertEqual(len({r["group_id"] for r in rows}), 28)
        self.assertEqual(sum(r["track"] == "unanswerable" for r in rows), 8)
        for i in range(0, len(rows), 2):
            left, right = rows[i:i + 2]
            self.assertEqual(left["group_id"], right["group_id"])
            self.assertEqual({d["id"]: d["text"] for d in left["candidates"]}, {d["id"]: d["text"] for d in right["candidates"]})
        validate_dataset(rows)

    def test_report_rejects_different_dataset_hashes(self):
        with self.assertRaises(ValueError):
            generate([dict(status="skipped", dataset_sha256="a"), dict(status="skipped", dataset_sha256="b")], [])

    def test_report_rejects_tampered_metrics(self):
        row = fixture()
        report = dict(schema_version=1, model="fake", status="complete", full_dataset=True, query_ids=["x"],
                      records=[dict(id="x", ranking=["0", "1", "2", "3"], scores={"0": 4, "1": 3, "2": 2, "3": 1}, metrics={"ndcg5": 0})])
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "fake.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_reports([temp], [row])


if __name__ == "__main__":
    unittest.main()

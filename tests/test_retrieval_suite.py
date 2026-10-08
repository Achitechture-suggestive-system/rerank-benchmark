"""Offline tests; fake model results test plumbing, never become published scores."""
import contextlib
import copy
import io
import json
import math
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.backends import Cohere
from benchmark.catalog import EMBEDDERS, RERANKERS
from benchmark.retrieval import DenseIndex, LexicalIndex, load_collection, retrieval_metrics, rrf_scores
from benchmark.run import ROOT, sha, write_json
from benchmark.suite import prepare, rerank_worker, retrieval_worker, validate_records
from benchmark.suite_report import build_report


def options(root):
    return SimpleNamespace(output=str(root), dataset=str(ROOT / "data/evidence_depth.jsonl"),
        resume=True, device="cpu", dtype="float32", batch_size=1, embedding_batch_size=4,
        max_length=1024, embedding_max_length=512, threads=4, seed=20260924,
        candidate_k=20, rrf_k=60, cutoff_layer=28, compress_ratio=2, compress_layers=[24],
        limit=0, pools=["bm25"], oracle=True, allow_large_models=False,
        trust_remote_code=False, local_files_only=False, cohere_min_interval=6.2,
        cohere_max_attempts=3, worker_method="bm25")


class RetrievalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(candidates=[dict(id="gold", grade=3), dict(id="partial", grade=2), dict(id="negative", grade=0)])

    def test_recall_and_idcg_use_all_positives_outside_top_k(self):
        metrics = retrieval_metrics(self.row, ["partial", "negative"], 2)
        self.assertAlmostEqual(metrics["ndcg@10"], 3 / (7 + 3 / math.log2(3)))
        self.assertEqual(metrics["candidate_recall"], .5)
        self.assertEqual(metrics["strict_candidate_recall"], 0)
        self.assertEqual(retrieval_metrics(self.row, ["negative"], 1)["ndcg@1"], 0)

    def test_hole_does_not_confuse_unjudged_and_judged_negative(self):
        metrics = retrieval_metrics(self.row, ["unjudged", "negative", "gold"])
        self.assertEqual(metrics["hole@1"], 1)
        self.assertEqual(metrics["hole@3"], 1 / 3)
        self.assertEqual(metrics["mrr@3"], 1 / 3)
        with self.assertRaises(ValueError):
            retrieval_metrics(self.row, ["gold", "gold"])

    def test_unanswerable_is_diagnostic_not_perfect_or_zero_quality(self):
        for doc in self.row["candidates"]:
            doc["grade"] = 0
        metrics = retrieval_metrics(self.row, ["gold", "unjudged"])
        self.assertNotIn("ndcg@10", metrics)
        self.assertNotIn("recall@10", metrics)
        self.assertEqual(metrics["hole@3"], .5)

    def test_collection_is_label_free_and_full_corpus(self):
        rows, corpus = load_collection(ROOT / "data/evidence_depth.jsonl")
        self.assertEqual((len(rows), len(corpus)), (56, 168))
        self.assertTrue(all(set(doc) == {"id", "text"} for doc in corpus))
        self.assertEqual([doc["id"] for doc in corpus], sorted(doc["id"] for doc in corpus))

    def test_collection_rejects_document_id_text_conflict(self):
        rows, _ = load_collection(ROOT / "data/evidence_depth.jsonl")
        rows[1]["candidates"][0]["text"] += " conflict"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Conflicting text"):
                load_collection(path)

    def test_bm25_idf_is_indexed_over_full_corpus(self):
        corpus = [dict(text="apple"), dict(text="boat"), dict(text="boat")]
        scores, _ = LexicalIndex(corpus, "bm25").score("apple")
        self.assertAlmostEqual(scores[0], math.log(1 + (3 - 1 + .5) / (1 + .5)))
        self.assertEqual(scores[1:], [0, 0])
        cosine, _ = LexicalIndex(corpus, "tfidf").score("apple")
        self.assertEqual(cosine, [1, 0, 0])

    def test_rrf_uses_one_based_ranks_not_raw_score_scales(self):
        scores = rrf_scores(["a", "b", "c"], [["a", "b", "c"], ["c", "b", "a"]])
        self.assertAlmostEqual(scores[0], 1 / 61 + 1 / 63)
        self.assertAlmostEqual(scores[1], 2 / 62)
        self.assertEqual(scores[0], scores[2])
        with self.assertRaises(ValueError):
            rrf_scores(["a", "b"], [["a"]])

    def test_e5_and_qwen_query_document_prefixes(self):
        index = DenseIndex.__new__(DenseIndex)
        index.spec = EMBEDDERS["e5-base"]
        self.assertEqual(index.format_text("xin chào", True), "query: xin chào")
        self.assertEqual(index.format_text("xin chào", False), "passage: xin chào")
        index.spec = EMBEDDERS["qwen-embed-0.6b"]
        self.assertTrue(index.format_text("hello", True).startswith("Instruct: "))
        self.assertEqual(index.format_text("hello", False), "hello")
        self.assertEqual(EMBEDDERS["minilm-multi"].max_tokens, 128)

    def test_catalog_ids_are_pinned_and_embedding_is_not_reranking(self):
        self.assertEqual((len(EMBEDDERS), len(RERANKERS)), (9, 15))
        for spec in list(EMBEDDERS.values()) + list(RERANKERS.values()):
            if spec.family != "cohere":
                self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
        self.assertNotEqual(EMBEDDERS["bge-m3-dense"].model_id, RERANKERS["bge-m3"].model_id)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.corpus = load_collection(ROOT / "data/evidence_depth.jsonl")
        self.output = tempfile.TemporaryDirectory()
        self.root = Path(self.output.name)
        self.args = options(self.root)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch("benchmark.suite.hardware", return_value={"test": True}))

    def tearDown(self):
        self.stack.close()
        self.output.cleanup()

    def retrieve(self):
        self.assertEqual(retrieval_worker(self.args, self.rows, self.corpus), 0)
        return self.root / "retrieval/bm25.json"

    def test_full_retrieval_resume_is_idempotent_and_partial_resumes(self):
        path = self.retrieve()
        original = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = sha(path)
        self.retrieve()
        self.assertEqual(sha(path), fingerprint)
        partial = copy.deepcopy(original)
        partial.update(status="failed", records=partial["records"][:2], completed_queries=2)
        write_json(path, partial)
        self.retrieve()
        resumed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(resumed["completed_queries"], 56)
        self.assertEqual(resumed["records"][:2], original["records"][:2])
        self.assertEqual(len(resumed["sessions"]), 2)

    def test_changed_config_or_code_must_not_mix_with_checkpoint(self):
        path = self.retrieve()
        fingerprint = sha(path)
        self.args.candidate_k = 10
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.retrieve()
        self.assertEqual(sha(path), fingerprint)
        self.args.candidate_k = 20
        with patch("benchmark.suite.source_hash", return_value="different"):
            with self.assertRaises(ValueError):
                prepare(path, "retrieval", "bm25", self.rows, self.corpus, self.args)

    def test_tampered_record_labels_scores_and_metrics_rejected(self):
        path = self.retrieve()
        report = json.loads(path.read_text(encoding="utf-8"))
        for mutation in (lambda r: r["metrics"].update({"ndcg@10": .123}),
                         lambda r: r.update(group_id="changed"),
                         lambda r: r["scores"].update(fake=99),
                         lambda r: r.update(latency_seconds=-1)):
            altered = copy.deepcopy(report)
            mutation(altered["records"][0])
            with self.assertRaises(ValueError):
                validate_records(altered, self.rows, self.corpus)

    def test_rerank_uses_retriever_pool_no_judgments_and_valid_report(self):
        path = self.retrieve()
        texts = {d["text"] for d in self.corpus}
        class FakeBackend:
            info = {"device": "test-double"}
            def score(self, query, docs):
                self_input = all(isinstance(doc, str) for doc in docs)
                if len(docs) != 2:  # Exclude warmup.
                    assert self_input and all(doc in texts for doc in docs)
                    assert len(docs) in (6, 20)
                return list(range(len(docs))), {"truncated_pairs": 0}
        self.args.worker_method = "bge-m3"
        with patch("benchmark.suite.make_backend", return_value=FakeBackend()):
            self.assertEqual(rerank_worker(self.args, self.rows, self.corpus), 0)
        build_report(self.root, self.rows, self.corpus, self.args.dataset)
        result = json.loads((self.root / "rerank/bge-m3/bm25.json").read_text(encoding="utf-8"))
        parent = json.loads(path.read_text(encoding="utf-8"))
        for record, base in zip(result["records"], parent["records"]):
            self.assertEqual(record["candidate_ids"], base["ranking"][:20])
            self.assertEqual(record["metrics"].get("candidate_recall"), base["metrics"].get("candidate_recall"))
        self.assertTrue((self.root / "comparisons.json").exists())
        self.assertTrue((self.root / "breakdown.csv").exists())
        parent["backend"]["tampered"] = True
        write_json(path, parent)
        with self.assertRaisesRegex(ValueError, "parent changed"):
            build_report(self.root, self.rows, self.corpus, self.args.dataset)

    def test_fusion_parent_changes_cannot_reuse_old_complete_result(self):
        bm25 = self.retrieve()
        fake_dense = self.root / "retrieval/e5-small.json"
        dense = json.loads(bm25.read_text(encoding="utf-8"))
        dense["method"] = "e5-small"
        write_json(fake_dense, dense)
        self.args.worker_method = "rrf-e5-small"
        self.assertEqual(retrieval_worker(self.args, self.rows, self.corpus), 0)
        dense["backend"]["changed"] = True
        write_json(fake_dense, dense)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            retrieval_worker(self.args, self.rows, self.corpus)
        with self.assertRaisesRegex(ValueError, "Fusion parent changed"):
            build_report(self.root, self.rows, self.corpus, self.args.dataset)

    def test_smoke_not_in_quality_table_missing_plan_is_visible(self):
        self.args.limit = 4
        self.assertEqual(retrieval_worker(self.args, self.rows[:4], self.corpus), 0)
        write_json(self.root / "plan.json", {"expected_artifacts": ["retrieval/bm25.json", "retrieval/e5-large.json"]})
        build_report(self.root, self.rows[:4], self.corpus, self.args.dataset)
        self.assertNotIn("bm25", (self.root / "summary.csv").read_text(encoding="utf-8-sig"))
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["missing_planned_artifacts"], ["retrieval/e5-large.json"])
        with self.assertRaisesRegex(ValueError, "Query selection mismatch"):
            build_report(self.root, self.rows, self.corpus, self.args.dataset)

    def test_missing_api_key_is_skipped_not_a_quality_result(self):
        self.retrieve()
        self.args.worker_method = "cohere-pro"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(rerank_worker(self.args, self.rows, self.corpus), 1)
        report = json.loads((self.root / "rerank/cohere-pro/oracle.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["records"], [])


class CohereRetryTests(unittest.TestCase):
    def test_429_backoff_pacing_and_index_mapping_without_real_sleep(self):
        with patch.dict(os.environ, COHERE_API_KEY="secret-never-publish"):
            model = Cohere("cohere-pro", SimpleNamespace(max_length=1024, cohere_min_interval=6.2, cohere_max_attempts=3))
        now = [0.0]
        payload = {"results": [{"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .1}]}
        attempts = [urllib.error.HTTPError("url", 429, "limited", {"Retry-After": "12"}, io.BytesIO(b"limited")), payload, payload]
        requests = []
        def send(request, timeout):
            requests.append(now[0])
            response = attempts.pop(0)
            if isinstance(response, Exception):
                raise response
            return io.BytesIO(json.dumps(response).encode())
        def sleep(seconds):
            now[0] += seconds
        with patch("benchmark.backends.time.monotonic", side_effect=lambda: now[0]), patch("benchmark.backends.time.perf_counter", side_effect=lambda: now[0]), patch("benchmark.backends.time.sleep", side_effect=sleep), patch("benchmark.backends.urllib.request.urlopen", side_effect=send):
            scores, audit = model.score("query", ["doc one", "doc two"])
            model.score("query", ["doc one", "doc two"])
        self.assertEqual(scores, [.1, .9])
        self.assertEqual(audit["attempts"], 2)
        self.assertEqual(audit["rate_limit_wait_seconds"], 12)
        self.assertAlmostEqual(requests[2] - requests[1], 6.2)

    def test_non_retryable_provider_error_redacts_key(self):
        with patch.dict(os.environ, COHERE_API_KEY="secret-never-publish"):
            model = Cohere("cohere-pro", SimpleNamespace(max_length=1024))
        error = urllib.error.HTTPError("url", 401, "unauthorized", {}, io.BytesIO(b"echo secret-never-publish"))
        with patch("benchmark.backends.urllib.request.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, r"HTTP 401.*REDACTED") as raised:
                model.score("q", ["d"])
        self.assertEqual(request.call_count, 1)
        self.assertNotIn("secret-never-publish", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

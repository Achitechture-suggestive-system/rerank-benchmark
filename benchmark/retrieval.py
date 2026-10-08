"""Full-corpus retrieval and evaluation against the original, sparse judgments."""
import collections
import hashlib
import json
import math
import re
from pathlib import Path

from .catalog import EMBEDDERS
from .metrics import validate_dataset

TASK = "Given a web search query, retrieve relevant passages that answer the query"
KS = (1, 3, 5, 10, 20, 50, 100)


def load_collection(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_dataset(rows)
    if not rows:
        raise ValueError("Empty dataset")
    texts = {}
    for row in rows:
        for doc in row["candidates"]:
            if doc["id"] in texts and texts[doc["id"]] != doc["text"]:
                raise ValueError(f"Conflicting text for document {doc['id']}")
            texts[doc["id"]] = doc["text"]
    # Query order and judgments never affect corpus order or model input.
    corpus = [{"id": key, "text": texts[key]} for key in sorted(texts)]
    return rows, corpus


def corpus_hash(corpus):
    return hashlib.sha256(json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class LexicalIndex:
    def __init__(self, corpus, method):
        self.method = method
        self.counts = [collections.Counter(tokenize(doc["text"])) for doc in corpus]
        self.df = collections.Counter(term for count in self.counts for term in count)
        self.n = len(corpus)
        self.lengths = [sum(count.values()) for count in self.counts]
        self.avg = sum(self.lengths) / self.n
        self.idf = {term: math.log((self.n + 1) / (freq + 1)) + 1 for term, freq in self.df.items()}
        self.vectors = [self.vector(count) for count in self.counts] if method == "tfidf" else None
        self.info = {"model_id": method, "device": "cpu", "idf_scope": "full_corpus",
                     "token_policy": "full_text_unicode_word_regex", "k1": 1.2, "b": .75}

    def vector(self, count):
        weighted = {term: (1 + math.log(freq)) * self.idf[term] for term, freq in count.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1
        return {term: value / norm for term, value in weighted.items()}

    def score(self, query):
        terms = collections.Counter(tokenize(query))
        if self.method == "tfidf":
            q = self.vector(terms)
            return [sum(value * doc.get(term, 0) for term, value in q.items()) for doc in self.vectors], {"truncated_pairs": 0}
        scores = []
        for doc, length in zip(self.counts, self.lengths):
            value = 0.0
            for term in terms:
                df, tf = self.df[term], doc[term]
                idf = math.log(1 + (self.n - df + .5) / (df + .5))
                value += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / max(self.avg, 1)))
            scores.append(value)
        return scores, {"truncated_pairs": 0}


def rrf_scores(ids, rankings, constant=60):
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    values = dict.fromkeys(ids, 0.0)
    for ranking in rankings:
        if len(ranking) != len(ids) or set(ranking) != set(ids):
            raise ValueError("RRF requires complete corpus permutations")
        for rank, key in enumerate(ranking, start=1):
            values[key] += 1 / (constant + rank)
    return [values[key] for key in ids]


def retrieval_metrics(row, ranking, candidate_k=20):
    """IDCG and recall denominators always use all judged positives, not top-K.

    Unjudged documents use zero gain as an explicit evaluation convention.
    Hole@K reports that missing-judgment exposure; it is not a relevance label.
    """
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate documents in ranking")
    grades = {doc["id"]: doc["grade"] for doc in row["candidates"]}
    relevant = {key for key, value in grades.items() if value >= 2}
    strict = {key for key, value in grades.items() if value == 3}
    ideal = sorted(grades.values(), reverse=True)
    gain = lambda gs: sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(gs))
    metrics = {}
    for k in KS:
        top = ranking[:k]
        metrics[f"hole@{k}"] = sum(key not in grades for key in top) / len(top) if top else 0.0
        if relevant:
            denom = gain(ideal[:k])
            metrics[f"ndcg@{k}"] = gain([grades.get(key, 0) for key in top]) / denom
            metrics[f"recall@{k}"] = len(relevant.intersection(top)) / len(relevant)
            metrics[f"mrr@{k}"] = next((1 / (i + 1) for i, key in enumerate(top) if key in relevant), 0.0)
    if relevant:
        metrics["hit@1"] = float(bool(ranking) and ranking[0] in relevant)
        metrics["strict@1"] = float(bool(ranking) and ranking[0] in strict)
        metrics["candidate_recall"] = len(relevant.intersection(ranking[:candidate_k])) / len(relevant)
        metrics["strict_candidate_recall"] = len(strict.intersection(ranking[:candidate_k])) / len(strict) if strict else None
    return metrics


class DenseIndex:
    def __init__(self, name, corpus, args):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.args, self.spec = torch, args, EMBEDDERS[name]
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: choose a GPU runtime and install CUDA-enabled torch")
        dtype = getattr(torch, args.dtype)
        self.cap = min(args.embedding_max_length, self.spec.max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id, revision=self.spec.revision,
                                                      local_files_only=args.local_files_only)
        if self.spec.family == "qwen-embed":
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(self.spec.model_id, revision=self.spec.revision,
                                              local_files_only=args.local_files_only,
                                              torch_dtype=dtype).to(args.device).eval()
        self.documents, index_audit = self.encode([doc["text"] for doc in corpus], query=False)
        self.info = dict(model_id=self.spec.model_id, model_revision=self.spec.revision, family=self.spec.family,
                         max_length=self.cap, requested_max_length=args.embedding_max_length,
                         pooling="last_valid_token" if self.spec.family == "qwen-embed" else "cls" if self.spec.family == "cls" else "masked_mean",
                         normalized=True, dtype=args.dtype, device=args.device, batch_size=args.embedding_batch_size,
                         query_instruction=TASK if self.spec.family == "qwen-embed" else None,
                         text_prefixes={"query": "query: ", "document": "passage: "} if self.spec.family == "e5" else None,
                         index_token_audit=index_audit, score_kind="cosine_similarity", bge_m3_mode="dense_only" if self.spec.family == "cls" else None)

    def format_text(self, text, query):
        if self.spec.family == "e5":
            return ("query: " if query else "passage: ") + text
        if self.spec.family == "qwen-embed" and query:
            return f"Instruct: {TASK}\nQuery:{text}"
        return text

    def encode(self, texts, query):
        torch, vectors = self.torch, []
        lengths, retained = [], []
        with torch.inference_mode():
            for start in range(0, len(texts), self.args.embedding_batch_size):
                batch = [self.format_text(text, query) for text in texts[start:start + self.args.embedding_batch_size]]
                full = self.tokenizer(batch, truncation=False)["input_ids"]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.cap, return_tensors="pt").to(self.args.device)
                extra = {"use_cache": False} if self.spec.family == "qwen-embed" else {}
                hidden = self.model(**inputs, **extra).last_hidden_state.float()
                mask = inputs["attention_mask"]
                if self.spec.family == "cls":
                    pooled = hidden[:, 0]
                elif self.spec.family == "qwen-embed":
                    pooled = hidden[:, -1]  # Left padding; the last position is always real.
                else:
                    pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1).clamp(min=1)
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
                lengths.extend(len(ids) for ids in full)
                retained.extend(mask.sum(1).cpu().tolist())
        if self.args.device == "cuda":
            torch.cuda.synchronize()
        return torch.cat(vectors), {"input_lengths": lengths, "retained_lengths": retained,
                                    "truncated_texts": sum(a > b for a, b in zip(lengths, retained))}

    def score(self, query):
        vector, audit = self.encode([query], query=True)
        return (vector @ self.documents.T)[0].tolist(), audit

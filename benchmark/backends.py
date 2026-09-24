"""Backends receive only query and document strings, never gold judgments."""
import collections
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

MODEL_IDS = {
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "qwen4b": "Qwen/Qwen3-Reranker-4B",
    "gemma": "BAAI/bge-reranker-v2.5-gemma2-lightweight",
    "cohere-pro": "rerank-v4.0-pro",
    "cohere-fast": "rerank-v4.0-fast",
    "bm25": "local-bm25-k1=1.2-b=0.75",
}
QWEN_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
GEMMA_PROMPT = "Predict whether passage B contains an answer to query A."


class BM25:
    info = {"model_id": MODEL_IDS["bm25"], "device": "cpu", "score_kind": "bm25", "token_policy": "full_text_unicode_word_regex"}

    def score(self, query, documents):
        tokenize = lambda s: re.findall(r"\w+", s.lower())
        docs = [collections.Counter(tokenize(d)) for d in documents]
        lengths = [sum(d.values()) for d in docs]
        avg = sum(lengths) / len(lengths)
        scores = []
        for doc, length in zip(docs, lengths):
            score = 0.0
            for term in set(tokenize(query)):
                df = sum(term in d for d in docs)
                idf = math.log(1 + (len(docs) - df + .5) / (df + .5))
                tf = doc[term]
                score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / max(avg, 1)))
            scores.append(score)
        return scores, {"truncated_pairs": 0, "input_lengths": lengths, "length_unit": "regex_words"}


def cohere_scores(payload, count):
    results = payload["results"]
    if len(results) != count or {r["index"] for r in results} != set(range(count)):
        raise ValueError("Cohere did not return each candidate exactly once")
    values = [None] * count
    for r in results:
        values[r["index"]] = float(r["relevance_score"])
    return values


class Cohere:
    def __init__(self, name, args):
        self.key = os.environ.get("COHERE_API_KEY")
        if not self.key:
            raise RuntimeError("COHERE_API_KEY is not configured")
        self.model = MODEL_IDS[name]
        self.max_length = args.max_length
        self.info = dict(model_id=self.model, device="hosted", score_kind="vendor_relevance_score",
                         token_policy="provider_max_tokens_per_doc; tokenizer and actual truncation not observable",
                         max_tokens_per_doc=self.max_length, provider_revision="not exposed by API")

    def score(self, query, documents):
        body = json.dumps(dict(model=self.model, query=query, documents=documents,
                               top_n=len(documents), max_tokens_per_doc=self.max_length)).encode()
        request = urllib.request.Request("https://api.cohere.com/v2/rerank", body,
                                         {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                return cohere_scores(payload, len(documents)), dict(truncated_pairs=None,
                    request_id=payload.get("id"), billed_units=payload.get("meta", {}).get("billed_units"), attempts=attempt + 1)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"Cohere HTTP {exc.code}") from None
                time.sleep(2 ** attempt)
        raise RuntimeError("Cohere retry limit reached")


class LocalModel:
    def __init__(self, name, args):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
        self.torch, self.name, self.args = torch, name, args
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the installed torch runtime")
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
        kw = {"local_files_only": args.local_files_only}
        if args.revision:
            kw["revision"] = args.revision
        if name == "gemma":
            if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
                raise ValueError("Gemma custom code requires --revision with an immutable 40-character commit SHA")
            # The author-supplied custom module imports internals removed in newer versions.
            major, minor = map(int, transformers.__version__.split(".")[:2])
            if major != 4 or not (42 <= minor <= 44):
                raise RuntimeError("Gemma custom code needs a separate transformers 4.42-4.44 environment; this adapter has not been inference-verified")
            kw.update(trust_remote_code=True, code_revision=args.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[name], **kw)
        factory = AutoModelForSequenceClassification if name == "bge-m3" else AutoModelForCausalLM
        self.model = factory.from_pretrained(MODEL_IDS[name], torch_dtype=dtype, **kw).to(args.device).eval()
        self.info = dict(model_id=MODEL_IDS[name], model_revision=getattr(self.model.config, "_commit_hash", None),
                         requested_revision=args.revision or "default/cached", device=args.device, dtype=args.dtype,
                         quantization="none", max_length=args.max_length, batch_size=args.batch_size,
                         score_kind="classification_logit" if name == "bge-m3" else "yes_minus_no_logit" if name == "qwen4b" else "layerwise_scalar_head",
                         prompt=QWEN_INSTRUCTION if name == "qwen4b" else GEMMA_PROMPT if name == "gemma" else None)
        if name == "qwen4b":
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.prefix = self.tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
            self.suffix = self.tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
            self.yes = self.tokenizer.encode("yes", add_special_tokens=False)
            self.no = self.tokenizer.encode("no", add_special_tokens=False)
            if len(self.yes) != 1 or len(self.no) != 1:
                raise ValueError("Expected single-token yes/no labels")
            self.info.update(prefix=QWEN_PREFIX, suffix=QWEN_SUFFIX)
        if name == "gemma":
            self.tokenizer.padding_side = "right"
            self.info.update(cutoff_layer=args.cutoff_layer, compress_ratio=args.compress_ratio,
                             compress_layers=args.compress_layers)

    def sync(self):
        if self.args.device == "cuda":
            self.torch.cuda.synchronize()

    def encode(self, query, documents):
        tok, cap = self.tokenizer, self.args.max_length
        truncated, lengths, retained, items, query_lens, prompt_lens = 0, [], [], [], [], []
        for doc in documents:
            if self.name == "bge-m3":
                full = tok(query, doc, truncation=False)["input_ids"]
                item = tok(query, doc, truncation="only_second", max_length=cap)
                size = len(full)
            elif self.name == "qwen4b":
                body = tok.encode(f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}", add_special_tokens=False)
                budget = cap - len(self.prefix) - len(self.suffix)
                query_only = tok.encode(f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: ", add_special_tokens=False)
                if budget < len(query_only):
                    raise ValueError("Context budget would truncate the query/instruction")
                ids = self.prefix + body[:budget] + self.suffix
                item = {"input_ids": ids, "attention_mask": [1] * len(ids)}
                size = len(self.prefix) + len(body) + len(self.suffix)
            else:
                sep = tok.encode("\n", add_special_tokens=False)
                query_ids = [tok.bos_token_id] + tok.encode("A: " + query, add_special_tokens=False) + sep
                passage = tok.encode("B: " + doc, add_special_tokens=False)
                prompt = sep + tok.encode(GEMMA_PROMPT, add_special_tokens=False)
                budget = cap - len(query_ids) - len(prompt)
                if budget < 1:
                    raise ValueError("Context budget cannot hold query and prompt")
                ids = query_ids + passage[:budget] + prompt
                item = {"input_ids": ids, "attention_mask": [1] * len(ids)}
                query_lens.append(len(query_ids))
                prompt_lens.append(len(prompt))
                size = len(query_ids) + len(passage) + len(prompt)
            lengths.append(size)
            retained.append(len(item["input_ids"]))
            truncated += int(size > len(item["input_ids"]))
            items.append(item)
        encoded = tok.pad(items, padding=True, return_tensors="pt").to(self.args.device)
        extras = {}
        if self.name == "gemma":
            extras = dict(cutoff_layers=[self.args.cutoff_layer], compress_ratio=self.args.compress_ratio,
                          compress_layer=self.args.compress_layers, query_lengths=query_lens, prompt_lengths=prompt_lens)
        return encoded, extras, dict(truncated_pairs=truncated, input_lengths=lengths, retained_lengths=retained, length_unit="model_tokens")

    def score(self, query, documents):
        scores, audit = [], {"truncated_pairs": 0, "input_lengths": [], "retained_lengths": [], "length_unit": "model_tokens"}
        with self.torch.inference_mode():
            for start in range(0, len(documents), self.args.batch_size):
                inputs, extras, stats = self.encode(query, documents[start:start + self.args.batch_size])
                if self.name == "bge-m3":
                    out = self.model(**inputs).logits.reshape(-1).float()
                elif self.name == "qwen4b":
                    logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                    out = logits[:, self.yes[0]] - logits[:, self.no[0]]
                else:
                    result = self.model(**inputs, use_cache=False, return_dict=True, **extras)
                    logits, mask = result.logits[0], result.attention_masks[0]
                    indices = mask.sum(dim=1).long() - 1
                    out = logits[self.torch.arange(logits.shape[0], device=logits.device), indices].reshape(-1).float()
                scores.extend(out.cpu().tolist())
                audit["truncated_pairs"] += stats["truncated_pairs"]
                audit["input_lengths"].extend(stats["input_lengths"])
                audit["retained_lengths"].extend(stats["retained_lengths"])
        self.sync()
        return scores, audit


def make_backend(name, args):
    if name == "bm25":
        return BM25()
    if name.startswith("cohere-"):
        return Cohere(name, args)
    return LocalModel(name, args)

"""Pure ranking metrics: no dependence on neural libraries or score scales."""
import math
import random
from collections import defaultdict
from statistics import mean


def validate_dataset(rows):
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate query ID")
        seen.add(row["id"])
        docs = row["candidates"]
        if len(docs) < 2 or len({d["id"] for d in docs}) != len(docs):
            raise ValueError("Invalid candidate IDs")
        if not row["query"].strip() or any(not d["text"].strip() for d in docs):
            raise ValueError("Empty model input")
        if any(type(d["grade"]) is not int or d["grade"] not in range(4) or not d["rationale"] for d in docs):
            raise ValueError("Invalid judgment")
        if (row["track"] == "unanswerable") != all(d["grade"] == 0 for d in docs):
            raise ValueError("Unanswerable track mismatch")


def rank_ids(ids, scores):
    if len(ids) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError("Expected one finite score per document")
    # Stable ties preserve the common shuffled candidate order.
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])]


def query_metrics(row, ranking, scores):
    grades = {d["id"]: d["grade"] for d in row["candidates"]}
    if len(ranking) != len(grades) or set(ranking) != set(grades):
        raise ValueError("Ranking must be a permutation of the candidate pool")
    if not any(g >= 2 for g in grades.values()):
        return None  # Undefined IDCG/MRR; never count an OOD query as perfect or zero.
    rel = [grades[d] for d in ranking]
    dcg = lambda gs: sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:5]))
    pairs = [(a, b) for a, ga in grades.items() for b, gb in grades.items() if ga == 3 and gb == 0]
    pair_acc = mean(1.0 if scores[a] > scores[b] else 0.5 if scores[a] == scores[b] else 0.0 for a, b in pairs) if pairs else None
    return dict(ndcg5=dcg(rel) / dcg(sorted(rel, reverse=True)),
                mrr5=next((1 / (i + 1) for i, g in enumerate(rel[:5]) if g >= 2), 0.0),
                hit1=float(rel[0] >= 2), strict1=float(rel[0] == 3),
                recall3=sum(g >= 2 for g in rel[:3]) / sum(g >= 2 for g in rel),
                hard_negative_win=pair_acc)


def aggregate(records):
    usable = [r for r in records if r["metrics"] is not None]
    return {"queries": len(usable), **{
        key: mean(r["metrics"][key] for r in usable if r["metrics"][key] is not None)
        for key in (usable[0]["metrics"] if usable else [])}}


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    x = (len(xs) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return xs[lo] + (xs[hi] - xs[lo]) * (x - lo)


def paired_comparison(baseline, challenger, metric="ndcg5", samples=2000):
    """Cluster bootstrap by scenario, keeping VI/EN paraphrases together."""
    a = {r["id"]: r for r in baseline if r["metrics"] is not None and r["track"] == "core"}
    b = {r["id"]: r for r in challenger if r["metrics"] is not None and r["track"] == "core"}
    if set(a) != set(b) or not a:
        raise ValueError("Paired comparison requires identical nonempty core query sets")
    groups = defaultdict(list)
    deltas = []
    for key in sorted(a):
        delta = b[key]["metrics"][metric] - a[key]["metrics"][metric]
        deltas.append(delta)
        groups[a[key]["group_id"]].append(delta)
    units = list(groups.values())
    rng = random.Random(20260924)
    boot = [mean(v for unit in rng.choices(units, k=len(units)) for v in unit) for _ in range(samples)]
    return dict(metric=metric, delta=mean(deltas), ci95=[percentile(boot, .025), percentile(boot, .975)],
                improved=sum(d > 1e-10 for d in deltas), harmed=sum(d < -1e-10 for d in deltas),
                unchanged=sum(abs(d) <= 1e-10 for d in deltas), groups=len(units), samples=samples)

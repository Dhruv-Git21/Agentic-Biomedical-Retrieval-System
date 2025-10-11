#!/usr/bin/env python3
import json, ast, math, argparse, requests, csv, time
from typing import Dict, List, Any
from itertools import islice

def parse_rel_map(val: Any) -> Dict[str, int]:
    if isinstance(val, dict):
        return {str(k): int(v) for k, v in val.items()}
    s = (val or "").strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return {str(k): int(v) for k, v in d.items()}
    except Exception:
        pass
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return {str(k): int(v) for k, v in d.items()}
    except Exception:
        pass
    return {}

def dcg(rels: List[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(retrieved: List[str], rel_map: Dict[str, int], k: int) -> float:
    if not rel_map:
        return 0.0
    gains = [rel_map.get(r, 0) for r in retrieved[:k]]
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(gains) / denom

def mrr_at_k(retrieved: List[str], rel_map: Dict[str, int], k: int) -> float:
    for i, rid in enumerate(retrieved[:k]):
        if rid in rel_map:
            return 1.0 / (i + 1)
    return 0.0

def precision_at_k(retrieved: List[str], rel_map: Dict[str, int], k: int) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in rel_map)
    denom = min(k, len(retrieved))
    return hits / max(1, denom)

def recall_at_k(retrieved: List[str], rel_map: Dict[str, int], k: int) -> float:
    if not rel_map:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in rel_map)
    return hits / len(rel_map)

def chunks(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

def main():
    ap = argparse.ArgumentParser(description="Evaluate Patient→Patient Retrieval via API")
    ap.add_argument("--json", required=True, help="PMC-Patients.json path")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--k", type=int, default=10, help="Top-K cutoff")
    ap.add_argument("--limit", type=int, default=2000, help="Max queries to evaluate (0=all)")
    ap.add_argument("--batch", type=int, default=64, help="Queries per /search_batch call")
    ap.add_argument("--out_csv", default="ppr_results.csv", help="Output CSV path")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect valid examples
    examples = []
    for d in data:
        q = (d.get("patient") or "").strip()
        uid = str(d.get("patient_uid") or "").strip()
        if q and uid:
            examples.append({
                "patient": q,
                "patient_uid": uid,
                "similar": parse_rel_map(d.get("similar_patients"))
            })
    if args.limit and args.limit < len(examples):
        examples = examples[:args.limit]

    print(f"Evaluating {len(examples)} queries at k={args.k} ...")

    sess = requests.Session()
    mrrs, ndcgs, precs, recs = [], [], [], []
    t0 = time.time()

    fieldnames = ["query_uid", "retrieved_ids", "relevant_ids",
                  f"MRR@{args.k}", f"nDCG@{args.k}", f"P@{args.k}", f"R@{args.k}"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        def call_batch(batch):
            payload = {
                "queries": [{"text": b["patient"], "exclude_id": b["patient_uid"]} for b in batch],
                "top_k": args.k, "final_k": args.k
            }
            r = sess.post(f"{args.api}/search_batch", json=payload, timeout=600)
            if r.status_code == 404:
                # fallback to single /search
                out = []
                for b in batch:
                    rr = sess.post(f"{args.api}/search", json={
                        "query": b["patient"], "top_k": args.k, "final_k": args.k, "exclude_id": b["patient_uid"]
                    }, timeout=600)
                    rr.raise_for_status()
                    out.append(rr.json().get("retrieved", []))
                return out
            r.raise_for_status()
            return r.json().get("results", [])

        done = 0
        total = len(examples)
        for batch in chunks(examples, args.batch):
            res_lists = call_batch(batch)
            for ex, res in zip(batch, res_lists):
                retrieved_ids = [str(x["id"]) for x in res]
                rel_map = ex["similar"]

                mrr = mrr_at_k(retrieved_ids, rel_map, args.k)
                ndcg = ndcg_at_k(retrieved_ids, rel_map, args.k)
                prec = precision_at_k(retrieved_ids, rel_map, args.k)
                rec = recall_at_k(retrieved_ids, rel_map, args.k)

                writer.writerow({
                    "query_uid": ex["patient_uid"],
                    "retrieved_ids": ";".join(retrieved_ids),
                    "relevant_ids": ";".join(rel_map.keys()),
                    f"MRR@{args.k}": f"{mrr:.6f}",
                    f"nDCG@{args.k}": f"{ndcg:.6f}",
                    f"P@{args.k}": f"{prec:.6f}",
                    f"R@{args.k}": f"{rec:.6f}",
                })

                mrrs.append(mrr); ndcgs.append(ndcg); precs.append(prec); recs.append(rec)
                done += 1
                if done % 50 == 0:
                    dt = time.time() - t0
                    print(f"[{done}/{total}] MRR@{args.k}={sum(mrrs)/done:.3f} "
                          f"nDCG@{args.k}={sum(ndcgs)/done:.3f} "
                          f"P@{args.k}={sum(precs)/done:.3f} "
                          f"R@{args.k}={sum(recs)/done:.3f}  ({dt:.1f}s)")

    n = len(mrrs) or 1
    print("\n=== FINAL (PPR) ===")
    print(f"Evaluated: {len(mrrs)}")
    print(f"MRR@{args.k}:   {sum(mrrs)/n:.4f}")
    print(f"nDCG@{args.k}:  {sum(ndcgs)/n:.4f}")
    print(f"Precision@{args.k}: {sum(precs)/n:.4f}")
    print(f"Recall@{args.k}:    {sum(recs)/n:.4f}")
    print(f"CSV: {args.out_csv}")

if __name__ == "__main__":
    main()

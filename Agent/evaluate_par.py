#!/usr/bin/env python3
import json, argparse, requests, csv, time
from typing import Dict, List

def mrr_at_k(titles: List[str], gold: str, k: int) -> float:
    for i, t in enumerate(titles[:k]):
        if t == gold:
            return 1.0 / (i + 1)
    return 0.0

def dcg(rels: List[int]) -> float:
    import math
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(titles: List[str], gold: str, k: int) -> float:
    gains = [1 if t == gold else 0 for t in titles[:k]]
    ideal = [1]  # only one relevant article
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(gains) / denom

def precision_at_k(titles: List[str], gold: str, k: int) -> float:
    # binary relevance, at most 1 relevant
    hits = 1 if gold in titles[:k] else 0
    denom = min(k, len(titles))
    return hits / max(1, denom)

def recall_at_k(titles: List[str], gold: str, k: int) -> float:
    # only one relevant article
    return 1.0 if gold in titles[:k] else 0.0

def main():
    ap = argparse.ArgumentParser(description="Evaluate Patient→Article Retrieval (PAR) via API")
    ap.add_argument("--json", required=True, help="PMC-Patients.json path")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--k", type=int, default=10, help="Top-K cutoff")
    ap.add_argument("--limit", type=int, default=2000, help="Max queries (0=all)")
    ap.add_argument("--batch", type=int, default=64, help="Queries per call to /search_par_batch")
    ap.add_argument("--out_csv", default="par_results.csv", help="Output CSV path")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect examples (patient text + its article title)
    examples = []
    for d in data:
        q = (d.get("patient") or "").strip()
        title = (d.get("title") or "").strip()
        if q and title:
            examples.append({"patient": q, "title": title})
    if args.limit and args.limit < len(examples):
        examples = examples[:args.limit]

    print(f"Evaluating {len(examples)} queries at k={args.k} ...")

    sess = requests.Session()
    mrrs, ndcgs, precs, recs, succs = [], [], [], [], []
    t0 = time.time()

    fieldnames = ["title_gold", "titles@k", f"MRR@{args.k}", f"nDCG@{args.k}", f"P@{args.k}", f"R@{args.k}", f"Success@{args.k}"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        def call_batch(batch):
            payload = {
                "queries": [{"text": b["patient"]} for b in batch],
                "top_k": args.k, "final_k": args.k
            }
            r = sess.post(f"{args.api}/search_par_batch", json=payload, timeout=600)
            r.raise_for_status()
            return r.json().get("results", [])

        done = 0
        total = len(examples)
        # chunk examples
        for i in range(0, len(examples), args.batch):
            batch = examples[i:i+args.batch]
            res_lists = call_batch(batch)
            for ex, res in zip(batch, res_lists):
                titles = [str(x.get("title") or "") for x in res]
                gold = ex["title"]

                mrr = mrr_at_k(titles, gold, args.k)
                ndcg = ndcg_at_k(titles, gold, args.k)
                prec = precision_at_k(titles, gold, args.k)
                rec = recall_at_k(titles, gold, args.k)
                succ = 1.0 if gold in titles[:args.k] else 0.0  # Success@K

                writer.writerow({
                    "title_gold": gold,
                    "titles@k": ";".join(titles),
                    f"MRR@{args.k}": f"{mrr:.6f}",
                    f"nDCG@{args.k}": f"{ndcg:.6f}",
                    f"P@{args.k}": f"{prec:.6f}",
                    f"R@{args.k}": f"{rec:.6f}",
                    f"Success@{args.k}": f"{succ:.6f}",
                })

                mrrs.append(mrr); ndcgs.append(ndcg); precs.append(prec); recs.append(rec); succs.append(succ)
                done += 1
                if done % 50 == 0:
                    dt = time.time() - t0
                    print(f"[{done}/{total}] MRR@{args.k}={sum(mrrs)/done:.3f} "
                          f"nDCG@{args.k}={sum(ndcgs)/done:.3f} "
                          f"P@{args.k}={sum(precs)/done:.3f} "
                          f"R@{args.k}={sum(recs)/done:.3f} "
                          f"Success@{args.k}={sum(succs)/done:.3f}  ({dt:.1f}s)")

    n = len(mrrs) or 1
    print("\n=== FINAL (PAR) ===")
    print(f"Evaluated: {len(mrrs)}")
    print(f"MRR@{args.k}:       {sum(mrrs)/n:.4f}")
    print(f"nDCG@{args.k}:      {sum(ndcgs)/n:.4f}")
    print(f"Precision@{args.k}: {sum(precs)/n:.4f}")
    print(f"Recall@{args.k}:    {sum(recs)/n:.4f}")
    print(f"Success@{args.k}:   {sum(succs)/n:.4f}")
    print(f"CSV: {args.out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json, argparse, requests
from typing import List, Dict

def post_rows(api: str, rows: List[Dict], text_col="patient", context_col="title", id_col="patient_uid",
              chunk_size=1200, chunk_overlap=200):
    payload = {
        "rows": rows,
        "text_col": text_col,
        "context_col": context_col,
        "id_col": id_col,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    r = requests.post(f"{api}/load_json", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to PMC-Patients.json")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--limit", type=int, default=0, help="Limit rows (0 = all)")
    ap.add_argument("--batch", type=int, default=5000, help="Rows per POST batch")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for d in data[: (args.limit or len(data))]:
        patient = (d.get("patient") or "").strip()
        uid = str(d.get("patient_uid") or "").strip()
        title = (d.get("title") or "").strip()
        if patient and uid:
            rows.append({"patient": patient, "patient_uid": uid, "title": title})

    print(f"Prepared {len(rows)} rows. Posting in batches of {args.batch} ...")
    i = 0
    while i < len(rows):
        chunk = rows[i:i+args.batch]
        resp = post_rows(args.api, chunk, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        print(f"[{i+1}-{i+len(chunk)}] -> {resp}")
        i += args.batch

if __name__ == "__main__":
    main()

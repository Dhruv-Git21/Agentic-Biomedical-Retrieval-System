# app.py
import io
from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from schemas import LoadJSONRequest, QueryRequest, QueryResponse, EvidenceItem
from document_store import DocumentStore
from retrieval import rerank_hybrid
from generation import answer_with_context, reflect_and_refine
from rag_config import CHUNK_SIZE, CHUNK_OVERLAP

app = FastAPI(title="Biomedical RAG API", version="1.0.0")

STORE = DocumentStore()
INDEX_READY = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": STORE.device,
        "models": STORE.mm.model_names,
    }

@app.post("/load_json")
def load_json(req: LoadJSONRequest):
    global INDEX_READY
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must be non-empty")
    df = pd.DataFrame(req.rows)
    stats = STORE.load_from_dataframe(
        df,
        text_col=req.text_col,
        context_col=req.context_col,
        id_col=req.id_col,
        chunk_size=req.chunk_size or CHUNK_SIZE,
        chunk_overlap=req.chunk_overlap or CHUNK_OVERLAP,
        append = True,
    )
    INDEX_READY = True
    return {"ok": True, "stats": stats}

@app.post("/load_csv")
def load_csv(file: UploadFile = File(...),
             text_col: str = "TEXT",
             context_col: str = "CONTEXT",
             id_col: str = "HADM_ID",
             chunk_size: int = CHUNK_SIZE,
             chunk_overlap: int = CHUNK_OVERLAP):
    global INDEX_READY
    if not file.filename.endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Only .csv or .tsv files supported.")
    content = file.file.read()
    sep = "," if file.filename.endswith(".csv") else "\t"
    try:
        df = pd.read_csv(io.BytesIO(content), sep=sep, dtype=str, keep_default_na=False, encoding_errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV/TSV: {e}")
    stats = STORE.load_from_dataframe(
        df,
        text_col=text_col,
        context_col=context_col,
        id_col=id_col,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        append = True,
    )
    INDEX_READY = True
    return {"ok": True, "stats": stats}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not INDEX_READY or STORE.mm.indices == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")
    # Retrieve final_k indices
    idxs = rerank_hybrid(STORE, req.query, top_k=req.top_k, final_k=req.final_k)
    if not idxs:
        return QueryResponse(answer="No relevant evidence found.", evidence=[], meta={"retrieved": 0})
    snippets = []
    ev = []
    for i in idxs:
        c = STORE.get_chunk(i)
        snippets.append(c.text)
        ev.append(EvidenceItem(text=c.text, hadm_id=c.hadm_id, context=c.context))
    # Generate
    draft = answer_with_context(req.query, snippets)
    refined = None
    if req.reflect:
        refined = reflect_and_refine(req.query, draft, snippets)
    return QueryResponse(answer=draft, refined_answer=refined, evidence=ev, meta={"retrieved": len(idxs)})

# ========= Retrieval-only endpoints for benchmarking =========
from typing import List, Optional
from fastapi import HTTPException
from pydantic import BaseModel

# Try to use retrieval with scores, fallback to indices-only
try:
    from retrieval import rerank_hybrid_with_scores  # returns [(idx, score), ...]
    _WITH_SCORES = True
except Exception:
    from retrieval import rerank_hybrid               # returns [idx, ...]
    _WITH_SCORES = False

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    final_k: int = 10
    exclude_id: Optional[str] = None  # prevent self-hit in PPR

class BatchQueryItem(BaseModel):
    text: str
    exclude_id: Optional[str] = None

class SearchBatchRequest(BaseModel):
    queries: List[BatchQueryItem]
    top_k: int = 10
    final_k: int = 10

# app.py

def _search_core(text: str, top_k: int, final_k: int):
    if _WITH_SCORES:
        return rerank_hybrid_with_scores(STORE, text, top_k=top_k, final_k=final_k)  # [(idx, score)]
    else:
        idxs = rerank_hybrid(STORE, text, top_k=top_k, final_k=final_k)
        return [(i, None) for i in idxs]

@app.post("/search")
def search(req: SearchRequest):
    global INDEX_READY, STORE
    if not INDEX_READY or getattr(STORE.mm, "indices", {}) == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")
    # *** Over-fetch to survive self-hit removal ***
    OF_MULT = 5
    idx_scores = _search_core(req.query, top_k=req.top_k * OF_MULT, final_k=req.final_k * OF_MULT)

    out = []
    for idx, score in idx_scores:
        c = STORE.get_chunk(idx)
        # normalize hadm_id to string
        cid = None if c.hadm_id is None else str(c.hadm_id)
        # exclude the query's own record (PPR)
        if req.exclude_id is not None and cid == str(req.exclude_id):
            continue
        out.append({
            "id": cid,
            "context": getattr(c, "context", None),
            "score": (float(score) if score is not None else None)
        })
        if len(out) >= req.final_k:
            break
    return {"retrieved": out, "params": {"top_k": req.top_k, "final_k": req.final_k}}

@app.post("/search_batch")
def search_batch(req: SearchBatchRequest):
    global INDEX_READY, STORE
    if not INDEX_READY or getattr(STORE.mm, "indices", {}) == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")

    OF_MULT = 5
    results = []
    for item in req.queries:
        idx_scores = _search_core(item.text, top_k=req.top_k * OF_MULT, final_k=req.final_k * OF_MULT)
        one = []
        for idx, score in idx_scores:
            c = STORE.get_chunk(idx)
            cid = None if c.hadm_id is None else str(c.hadm_id)
            if item.exclude_id is not None and cid == str(item.exclude_id):
                continue
            one.append({
                "id": cid,
                "context": getattr(c, "context", None),
                "score": (float(score) if score is not None else None)
            })
            if len(one) >= req.final_k:
                break
        results.append(one)
    return {"results": results, "params": {"top_k": req.top_k, "final_k": req.final_k}}

# =============================================================

# ===== Patient -> Article Retrieval (PAR) =====
from pydantic import BaseModel
from typing import Optional, List

class SearchParRequest(BaseModel):
    text: str
    top_k: int = 10
    final_k: int = 10

class SearchParBatchItem(BaseModel):
    text: str

class SearchParBatchRequest(BaseModel):
    queries: List[SearchParBatchItem]
    top_k: int = 10
    final_k: int = 10

def _search_core(text: str, top_k: int, final_k: int):
    # Reuse existing reranker, prefer with-scores if available:contentReference[oaicite:1]{index=1}
    if _WITH_SCORES:
        return rerank_hybrid_with_scores(STORE, text, top_k=top_k, final_k=final_k)  # [(idx, score)]
    else:
        idxs = rerank_hybrid(STORE, text, top_k=top_k, final_k=final_k)
        return [(i, None) for i in idxs]

@app.post("/search_par")
def search_par(req: SearchParRequest):
    global INDEX_READY, STORE
    if not INDEX_READY or getattr(STORE.mm, "indices", {}) == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")

    OF_MULT = 5
    idx_scores = _search_core(req.text, top_k=req.top_k * OF_MULT, final_k=req.final_k * OF_MULT)

    # group by article title (context), keep best score per title
    best_by_title = {}
    for idx, score in idx_scores:
        c = STORE.get_chunk(idx)
        title = getattr(c, "context", None) or ""
        if not title:
            continue
        sc = float(score) if score is not None else 0.0
        if title not in best_by_title or sc > best_by_title[title]["score"]:
            best_by_title[title] = {
                "title": title,
                "score": sc,
                "example_id": None if c.hadm_id is None else str(c.hadm_id),
            }

    # top final_k articles
    arts = sorted(best_by_title.values(), key=lambda x: x["score"], reverse=True)[: req.final_k]
    return {"articles": arts, "params": {"top_k": req.top_k, "final_k": req.final_k}}

@app.post("/search_par_batch")
def search_par_batch(req: SearchParBatchRequest):
    global INDEX_READY, STORE
    if not INDEX_READY or getattr(STORE.mm, "indices", {}) == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")

    OF_MULT = 5
    results = []
    for item in req.queries:
        idx_scores = _search_core(item.text, top_k=req.top_k * OF_MULT, final_k=req.final_k * OF_MULT)
        best_by_title = {}
        for idx, score in idx_scores:
            c = STORE.get_chunk(idx)
            title = getattr(c, "context", None) or ""
            if not title:
                continue
            sc = float(score) if score is not None else 0.0
            if title not in best_by_title or sc > best_by_title[title]["score"]:
                best_by_title[title] = {
                    "title": title,
                    "score": sc,
                    "example_id": None if c.hadm_id is None else str(c.hadm_id),
                }
        arts = sorted(best_by_title.values(), key=lambda x: x["score"], reverse=True)[: req.final_k]
        results.append(arts)
    return {"results": results, "params": {"top_k": req.top_k, "final_k": req.final_k}}
# =============================================


# In app.py
from fastapi import FastAPI
from pydantic import BaseModel

# ... existing imports and STORE setup ...

@app.get("/stats")
def stats():
    """
    Returns index-level stats: number of chunks, and index sizes per model.
    """
    try:
        info = {
            "chunks": len(getattr(STORE, "chunks", []) or []),
            "models": [],
        }
        mm = getattr(STORE, "mm", None)
        if mm is not None:
            names = getattr(mm, "model_names", [])
            idxs = getattr(mm, "indices", {})
            dims = getattr(mm, "dim_by_model", {})
            for name in names:
                index = idxs.get(name)
                nvec = int(getattr(index, "ntotal", 0)) if index is not None else 0
                info["models"].append({"name": name, "dim": dims.get(name), "vectors": nvec})
        return info
    except Exception as e:
        return {"error": str(e)}


# app.py (append near other endpoints)
from fastapi import UploadFile, File
import pandas as pd, json, re
from typing import List, Dict, Any

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\.\,\!\?\:\;\-\_\(\)\[\]\"'/\\]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _f1(pred: str, gold: str) -> float:
    pt, gt = _norm(pred).split(), _norm(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt:  return 0.0
    from collections import Counter
    c = Counter(pt); match = 0
    for t in gt:
        if c[t] > 0: match += 1; c[t] -= 1
    if match == 0: return 0.0
    prec, rec = match/len(pt), match/len(gt)
    return 2*prec*rec/(prec+rec)

def _em(pred: str, gold: str) -> float:
    return 1.0 if _norm(pred) == _norm(gold) else 0.0

@app.post("/eval_pubmedqa")
def eval_pubmedqa(file: UploadFile = File(...),
                  q_col: str = "question",
                  a_col: str = "answer",
                  choices_col: str = "",
                  top_k: int = 16,
                  final_k: int = 5,
                  reflect: bool = True):
    if not INDEX_READY or getattr(STORE.mm, "indices", {}) == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")

    content = file.file.read()
    name = (file.filename or "").lower()
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        rows = [json.loads(l) for l in content.decode("utf-8").splitlines() if l.strip()]
        df = pd.DataFrame(rows)
    elif name.endswith(".tsv") or name.endswith(".txt"):
        from io import BytesIO
        df = pd.read_csv(BytesIO(content), sep="\t", dtype=str, keep_default_na=False, encoding_errors="ignore")
    else:
        from io import BytesIO
        df = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False, encoding_errors="ignore")

    q_col = q_col if q_col in df.columns else list(df.columns)[0]
    a_col = a_col if a_col in df.columns else None
    choices = (choices_col if choices_col in df.columns else None)

    results: List[Dict[str, Any]] = []
    n = em_sum = f1_sum = acc_sum = 0.0

    for _, row in df.iterrows():
        q = str(row.get(q_col, "")).strip()
        gold = str(row.get(a_col, "")).strip() if a_col else ""
        opts = row.get(choices, "") if choices else ""
        if opts:
            try:
                parsed = json.loads(opts)
                if isinstance(parsed, list):
                    q = q + "\nOptions: " + ", ".join(parsed)
            except Exception:
                parts = [x.strip() for x in str(opts).split(";") if x.strip()]
                if parts:
                    q = q + "\nOptions: " + ", ".join(parts)

        idxs = rerank_hybrid(STORE, q, top_k=top_k, final_k=final_k)
        snippets = [STORE.get_chunk(i).text for i in idxs]
        draft = answer_with_context(q, snippets)
        pred = reflect_and_refine(q, draft, snippets) if reflect else draft

        em = _em(pred, gold) if gold else 0.0
        f1 = _f1(pred, gold) if gold else 0.0

        acc = 0.0
        if gold:
            pnorm = _norm(pred)
            label = "yes" if " yes" in (" " + pnorm) else ("no" if " no" in (" " + pnorm) else ("maybe" if "maybe" in pnorm else ""))
            if label and _norm(gold) == label: acc = 1.0

        n += 1; em_sum += em; f1_sum += f1; acc_sum += acc
        results.append({"question": q, "gold": gold, "pred": pred, "em": em, "f1": f1, "acc_YNM": acc})

    return {
        "count": int(n),
        "avg_EM": (em_sum / max(1, n)),
        "avg_F1": (f1_sum / max(1, n)),
        "avg_Acc_YNM": (acc_sum / max(1, n)),
        "rows": results[:200]  # sample in response; write CSV client-side if needed
    }


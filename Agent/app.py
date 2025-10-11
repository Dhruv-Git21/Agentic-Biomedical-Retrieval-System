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
        chunk_overlap=req.chunk_overlap or CHUNK_OVERLAP
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
        chunk_overlap=chunk_overlap
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
    idx_scores = _search_core(req.query, req.top_k, req.final_k)

    out = []
    for idx, score in idx_scores:
        c = STORE.get_chunk(idx)
        # exclude the query's own record (PPR)
        if req.exclude_id is not None and str(c.hadm_id) == str(req.exclude_id):
            continue
        out.append({
            "id": c.hadm_id,
            "context": getattr(c, "context", None),
            "score": (float(score) if score is not None else None)
        })
    return {"retrieved": out, "params": {"top_k": req.top_k, "final_k": req.final_k}}

@app.post("/search_batch")
def search_batch(req: SearchBatchRequest):
    global INDEX_READY, STORE
    if not INDEX_READY or getattr(STORE.mm, "indices", {}) == {}:
        raise HTTPException(status_code=400, detail="No index loaded. Call /load_json or /load_csv first.")
    results = []
    for item in req.queries:
        idx_scores = _search_core(item.text, req.top_k, req.final_k)
        one = []
        for idx, score in idx_scores:
            c = STORE.get_chunk(idx)
            if item.exclude_id is not None and str(c.hadm_id) == str(item.exclude_id):
                continue
            one.append({
                "id": c.hadm_id,
                "context": getattr(c, "context", None),
                "score": (float(score) if score is not None else None)
            })
        results.append(one)
    return {"results": results, "params": {"top_k": req.top_k, "final_k": req.final_k}}
# =============================================================



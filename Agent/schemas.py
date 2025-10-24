# schemas.py
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class LoadJSONRequest(BaseModel):
    rows: List[Dict[str, Any]]
    text_col: str = Field(default="TEXT")
    context_col: Optional[str] = Field(default="CONTEXT")
    id_col: Optional[str] = Field(default="HADM_ID")
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 12
    final_k: int = 5
    reflect: bool = False
    # --- Agentic RAG controls ---
    agentic: bool = False                 # turn the multi-hop loop on/off
    max_loops: int = 3                    # cap retrieval->answer cycles
    min_new_evidence: int = 2             # stop if fewer novel chunks are found
    stop_on_confident: bool = True        # allow judge to halt early if confident
    confidence_threshold: float = 0.65    # 0..1; higher = stricter stopping

class EvidenceItem(BaseModel):
    text: str
    hadm_id: Optional[Any] = None
    context: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    refined_answer: Optional[str] = None
    evidence: List[EvidenceItem]
    meta: Dict[str, Any] = {}

# --- schemas.py (ADD) -------------------------------------------------------
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class TraceRetrievalView(BaseModel):
    query: str
    expanded_terms: List[str]
    per_model: Dict[str, Dict[str, List[float]]]  # indices, scores_raw, scores_norm
    aggregated_embed: Dict[int, float]            # doc_idx -> aggregated embedding score (0..1)
    hybrid_breakdown: Dict[int, Dict[str, float]] # doc_idx -> {embed_score, keyword_overlap, exact_phrase_bonus, offtopic_penalty, final_score}
    mmr_selected_doc_indices: List[int]
    selected_evidence: List[Dict[str, Any]]       # [{index, hadm_id, context, text_preview, score}]

class TraceAgentView(BaseModel):
    confidence: float
    need_more: bool
    reasons: str
    sub_queries: List[str]

class TraceStep(BaseModel):
    step: int
    retrieval: TraceRetrievalView
    agent: Optional[TraceAgentView] = None
    stop_reason: Optional[str] = None  # e.g., confident, no_results, insufficient_novel_evidence, no_refinements, max_step_or_agent_off

class TraceQueryRequest(BaseModel):
    query: str
    top_k: int = 12
    final_k: int = 5
    max_loops: int = 3
    min_new_evidence: int = 2
    stop_on_confident: bool = True
    confidence_threshold: float = 0.65
    reflect: bool = False

class TraceQueryResponse(BaseModel):
    question: str
    steps: List[TraceStep]
    final_answer: str
    refined_answer: Optional[str] = None
    final_evidence: List[EvidenceItem]
    final_confidence: Optional[float] = None
    stop_reason: Optional[str] = None
# ---------------------------------------------------------------------------

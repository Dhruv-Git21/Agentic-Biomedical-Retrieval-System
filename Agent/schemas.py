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

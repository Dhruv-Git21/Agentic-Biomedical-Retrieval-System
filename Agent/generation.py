# generation.py
from typing import List
from openai import OpenAI
from rag_config import OPENAI_API_KEY, DEFAULT_LLM_MODEL
import json
import os
import logging
logger = logging.getLogger("agentic")

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    
    # Pull from rag_config first, then env (belt + suspenders)
    from rag_config import OPENAI_API_KEY as CFG_KEY
    key = (CFG_KEY or os.getenv("OPENAI_API_KEY", "REPLACE WITH YOUR API KEY")).strip()
    
    # Log source without leaking the key
    src = "rag_config" if CFG_KEY else ("env" if key else "unset")
    logger.info("get_client(): OPENAI key source=%s len=%d", src, len(key))
    
    if not key:
        raise RuntimeError("OpenAI API key not set. Set env OPENAI_API_KEY or define a fallback in rag_config.py.")
    
    # Create client with explicit key (don’t rely on process env)
    _client = OpenAI(api_key=key)
    return _client


def _build_context(snippets: List[str]) -> str:
    # small, efficient format; avoids long prompts
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))

def answer_with_context(question: str, snippets: List[str]) -> str:
    client = get_client()
    system = (
        "You are a careful biomedical assistant. "
        "Use ONLY the supplied evidence to answer. "
        "Cite snippets with [number]. If evidence is insufficient, say so."
    )
    user = f"Question: {question}\n\nEvidence:\n{_build_context(snippets)}\n\nAnswer:"
    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def reflect_and_refine(question: str, draft: str, snippets: List[str]) -> str:
    client = get_client()
    system = (
        "You are a strict biomedical verifier. Improve the draft only if needed and ensure all claims are supported by evidence."
    )
    user = (
        f"Question: {question}\n"
        f"Evidence:\n{_build_context(snippets)}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Instructions:\n"
        "- Verify support for each claim.\n"
        "- If something is missing, add it using the evidence.\n"
        "- Keep it concise and cite snippets like [1], [2].\n"
        "- If evidence is insufficient, say so.\n\n"
        "Refined answer:"
    )
    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.1,
    )
    return resp.choices[0].message.content

# --------------------------- Agentic helpers ---------------------------
def agent_judge_and_refine(
    question: str,
    running_answer: str,
    evidence_snippets: List[str],
    confidence_threshold: float = 0.65,
    max_new_queries: int = 3,
) -> dict:
    """
    Returns structured JSON:
      {
        "confidence": float in [0,1],
        "need_more": bool,
        "reasons": str,
        "sub_queries": [str, ...]    # <= max_new_queries
      }
    """
    client = get_client()
    system = (
        "You are a biomedical RAG controller. Judge coverage and propose targeted, "
        "non-overlapping follow-up search queries ONLY if needed. Respond in strict JSON."
    )
    user = (
        "Question:\n"
        f"{question}\n\n"
        "Current draft answer:\n"
        f"{running_answer}\n\n"
        "Evidence snippets (numbered):\n"
        + "\n".join(f"[{i+1}] {s}" for i, s in enumerate(evidence_snippets[:12])) + "\n\n"
        "Decide:\n"
        f"- confidence: 0..1 for whether the current draft sufficiently answers the question.\n"
        f"- need_more: true if additional retrieval would materially improve answer.\n"
        f"- reasons: short rationale.\n"
        f"- sub_queries: up to {max_new_queries} concrete search queries that would fill the gaps if need_more is true.\n"
        "Return STRICT JSON with keys: confidence, need_more, reasons, sub_queries."
    )
    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        data = json.loads(raw[start:end+1]) if start != -1 and end != -1 else {
            "confidence": 0.0, "need_more": False, "reasons": "Parser fallback", "sub_queries": []
        }
    conf = float(data.get("confidence", 0.0))
    need_more = bool(data.get("need_more", False))
    subq = [q.strip() for q in (data.get("sub_queries") or []) if q and q.strip()]
    return {
        "confidence": conf,
        "need_more": (conf < confidence_threshold) and need_more,
        "reasons": data.get("reasons", ""),
        "sub_queries": subq[:max_new_queries],
    }

def squash_queries(original: str, refinements: List[str], max_len: int = 512) -> str:
    """
    Combine original with a small set of sub-queries in a compact suffix that
    plays nicely with your multi-model + keyword heuristics.
    """
    if not refinements:
        return original
    top_refs = [r.strip() for r in refinements[:2] if r.strip()]
    joined = f"{original} ;; ASPECTS: " + " | ".join(top_refs) if top_refs else original
    return joined[:max_len]

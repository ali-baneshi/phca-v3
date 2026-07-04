"""Phase 14: replay-safe action explainability from JSONL action_rationale."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from phca.monitoring.cognitive_panels import classify_action_mechanism
from phca.monitoring.observability import ObservabilityFrame

_DRIVE_LABELS = {
    1: "D1 PredErr",
    2: "D2 Critical",
    3: "D3 Compet",
    4: "D4 Curious",
    5: "D5 Energy",
    6: "D6 Empower",
}

DECISION_REASONS = (
    "explore",
    "d5_stay",
    "rbta_safe",
    "greedy_fallback",
    "prediction",
    "continuous_mpc",
    "continuous_explore",
)


def infer_decision_reason(rationale: Dict[str, Any]) -> str:
    """Backfill decision_reason from legacy action_rationale fields."""
    r = rationale or {}
    explicit = r.get("decision_reason")
    if isinstance(explicit, str) and explicit:
        return explicit
    if r.get("rbta_safe_mode"):
        return "rbta_safe"
    if r.get("explored"):
        return "continuous_explore" if r.get("continuous") else "explore"
    if r.get("goal_id") == 5 or r.get("note") == "D5 energy: STAY":
        return "d5_stay"
    if r.get("greedy_fallback"):
        return "greedy_fallback"
    if r.get("continuous"):
        return "continuous_mpc"
    if r.get("best_score") is not None or r.get("k_candidates"):
        return "prediction"
    return "prediction"


def explain_fields_present(rationale: Dict[str, Any]) -> bool:
    """True when Phase 14 explain keys are present."""
    r = rationale or {}
    return bool(
        r.get("decision_reason")
        or r.get("mechanism")
        or r.get("drive_id") is not None
    )


def _drive_line(f: ObservabilityFrame, r: Dict[str, Any]) -> str:
    did = r.get("drive_id") or r.get("goal_id") or getattr(f, "active_drive_id", None)
    if not did:
        return "drive —"
    lbl = _DRIVE_LABELS.get(int(did), f"D{did}")
    lvls = list(getattr(f, "drive_levels", []) or [])
    lvl_s = ""
    if lvls and 0 < int(did) <= len(lvls):
        lvl_s = f" (lvl {lvls[int(did) - 1]:.2f})"
    return f"drive {lbl}{lvl_s}"


def _score_margin(scores: List[float]) -> Optional[float]:
    if len(scores) < 2:
        return None
    ordered = sorted(float(x) for x in scores)
    return ordered[-1] - ordered[-2]


def _chosen_label(f: ObservabilityFrame, r: Dict[str, Any], scores: List[float]) -> str:
    if r.get("chosen_label"):
        return str(r["chosen_label"])
    names = list(getattr(f, "action_names", []) or [])
    ci = r.get("chosen_idx")
    is_cont = bool(r.get("continuous", getattr(f, "continuous_action", None) is not None))
    if is_cont:
        tau = getattr(f, "continuous_action", None)
        if tau is not None:
            arr = np.asarray(tau, dtype=np.float32).reshape(-1)
            if arr.size:
                parts = ", ".join(f"{v:.2f}" for v in arr[:4])
                return f"τ=[{parts}]"
        if isinstance(ci, (int, float)) and int(ci) >= 0:
            return f"τ#{int(ci)}"
        return "τ"
    if isinstance(ci, (int, float)):
        idx = int(ci)
        if 0 <= idx < len(names) and names[idx]:
            return f"{names[idx]} #{idx}"
        return f"#{idx}"
    if scores:
        idx = int(np.argmax(scores))
        if 0 <= idx < len(names) and names[idx]:
            return f"{names[idx]} #{idx}"
        return f"#{idx}"
    return "—"


def build_explain_chain(f: ObservabilityFrame) -> List[str]:
    """Causal chain from drive → scoring → chosen action (JSONL fields only)."""
    r = dict(getattr(f, "action_rationale", {}) or {})
    if not r:
        return []
    reason = infer_decision_reason(r)
    mech = r.get("mechanism") or classify_action_mechanism(r)
    scores = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
    lines: List[str] = [_drive_line(f, r)]
    eps = r.get("eps")
    eps_s = f"{float(eps):.3f}" if isinstance(eps, (int, float)) else "—"
    k = r.get("k_candidates")
    k_s = str(int(k)) if isinstance(k, (int, float)) else "—"
    lines.append(f"reason {reason} · mech {mech} · k={k_s} ε={eps_s}")
    bs = r.get("best_score")
    if isinstance(bs, (int, float)):
        margin = _score_margin(scores)
        margin_s = f" · margin {margin:+.3f}" if margin is not None else ""
        lines.append(f"score {float(bs):.3f}{margin_s}")
    comps = r.get("score_components") or {}
    if comps:
        top = ", ".join(f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in list(comps.items())[:4])
        lines.append(f"components {top}")
    chosen = _chosen_label(f, r, scores)
    lines.append(f"chosen {chosen}")
    facts = r.get("relevant_facts_summary") or []
    if facts:
        ids = ", ".join(str(x.get("fact_id", "?")) for x in facts[:3])
        extra = len(facts) - 3
        if extra > 0:
            ids += f" +{extra}"
        lines.append(f"facts {len(facts)} relevant ({ids})")
    elif r.get("relevant_fact_ids"):
        ids = ", ".join(str(x) for x in r["relevant_fact_ids"][:3])
        lines.append(f"facts {len(r['relevant_fact_ids'])} ids ({ids})")
    return lines


def format_explain_block(f: ObservabilityFrame, *, max_lines: int = 5) -> str:
    chain = build_explain_chain(f)
    if not chain:
        r = getattr(f, "action_rationale", {}) or {}
        if r and not explain_fields_present(r):
            return "explain: (legacy rationale — no decision_reason)"
        return ""
    return "\n".join(chain[:max_lines])


def explain_anchor_bundle(f: ObservabilityFrame) -> Dict[str, Any]:
    r = dict(getattr(f, "action_rationale", {}) or {})
    chain = build_explain_chain(f)
    scores = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
    return {
        "cycle_id": int(getattr(f, "cycle_id", 0) or 0),
        "chain": chain,
        "decision_reason": infer_decision_reason(r),
        "mechanism": r.get("mechanism") or classify_action_mechanism(r),
        "chosen_label": _chosen_label(f, r, scores),
    }


def explain_smoke_pass(frames: List[ObservabilityFrame]) -> bool:
    """True when first/mid/last frames have Phase 14 explain fields."""
    if not frames:
        return False
    indices = {0, len(frames) // 2, len(frames) - 1}
    for i in indices:
        r = getattr(frames[i], "action_rationale", {}) or {}
        if not (r.get("decision_reason") or r.get("mechanism")):
            return False
    return True

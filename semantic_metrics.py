import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG: dict[str, Any] = {
    "top_n_transitions": 7,
    "top_n_outliers": 7,
    "top_n_issues": 12,
    "redundancy_threshold": 0.90,
    "coverage_threshold": 0.72,
    "novelty_low_threshold": 0.58,
    "novelty_high_threshold": 0.88,
    "adj_weak_z": -1.0,
    "outlier_z": -1.0,
}


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=np.float32)
    iu = np.triu_indices_from(matrix, k=1)
    return matrix[iu].astype(np.float32)


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    return vectors @ vectors.T


def analyze_document_metrics(
    meta_path: str | Path,
    emb_path: str | Path,
    out_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    meta_path = Path(meta_path)
    emb_path = Path(emb_path)
    out_path = Path(out_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    emb = np.load(emb_path).astype(np.float32)
    n_segments = len(meta)
    if emb.shape[0] != n_segments:
        raise ValueError(f"Mismatch: emb={emb.shape[0]} meta={n_segments}")

    s_matrix = _cosine_matrix(emb)
    roles = np.array([m.get("role", "") for m in meta])
    paths = [str(m.get("path", "")) for m in meta]
    parents = np.array([str(m.get("parent", "")) for m in meta])

    main_idx = np.where(roles == "main")[0]
    intro_idx = np.where(roles == "intro")[0]
    conc_idx = np.where(roles == "conclusion")[0]

    intro_i = int(intro_idx[0]) if intro_idx.size else None
    conc_i = int(conc_idx[0]) if conc_idx.size else None

    adj = np.array([float(s_matrix[i, i + 1]) for i in range(n_segments - 1)], dtype=np.float32)
    adj_mean = float(adj.mean()) if adj.size else float("nan")
    adj_std = float(adj.std(ddof=0)) if adj.size else float("nan")
    adj_z = (adj - adj_mean) / adj_std if np.isfinite(adj_std) and adj_std > 0 else np.zeros_like(adj)

    weak_transition_indices = [int(i) for i, z in enumerate(adj_z) if float(z) <= cfg["adj_weak_z"]]
    weakest_order = np.argsort(adj)[: min(int(cfg["top_n_transitions"]), adj.size)] if adj.size else np.array([], dtype=np.int64)

    weakest_transitions: list[dict[str, Any]] = []
    for i in weakest_order:
        i = int(i)
        weakest_transitions.append(
            {
                "from_index": i,
                "to_index": i + 1,
                "from_path": paths[i],
                "to_path": paths[i + 1],
                "adj_similarity": float(adj[i]),
                "adj_z": float(adj_z[i]),
            }
        )

    main_main_mean = float("nan")
    intro_to_main_mean = float("nan")
    conc_to_main_mean = float("nan")
    intro_delta = float("nan")
    conc_delta = float("nan")

    if main_idx.size:
        mm_vals = _upper_triangle_values(s_matrix[np.ix_(main_idx, main_idx)])
        if mm_vals.size:
            main_main_mean = float(mm_vals.mean())

    if intro_i is not None and main_idx.size:
        intro_to_main_mean = float(s_matrix[intro_i, main_idx].mean())
    if conc_i is not None and main_idx.size:
        conc_to_main_mean = float(s_matrix[conc_i, main_idx].mean())

    if np.isfinite(intro_to_main_mean) and np.isfinite(main_main_mean):
        intro_delta = float(intro_to_main_mean - main_main_mean)
    if np.isfinite(conc_to_main_mean) and np.isfinite(main_main_mean):
        conc_delta = float(conc_to_main_mean - main_main_mean)

    redund_90 = float("nan")
    if main_idx.size >= 2:
        mm_vals = _upper_triangle_values(s_matrix[np.ix_(main_idx, main_idx)])
        if mm_vals.size:
            redund_90 = float(np.mean(mm_vals > cfg["redundancy_threshold"]))

    step_vecs = emb[1:] - emb[:-1]
    step_norms = np.linalg.norm(step_vecs, axis=1) if step_vecs.size else np.array([], dtype=np.float32)

    turn_sharpness = []
    for i in range(len(step_vecs) - 1):
        a = step_vecs[i]
        b = step_vecs[i + 1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        turn_sharpness.append(1.0 - float(np.dot(a, b) / denom))
    turn_sharpness = np.array(turn_sharpness, dtype=np.float32)

    step_mean = float(step_norms.mean()) if step_norms.size else 0.0
    turn_mean = float(turn_sharpness.mean()) if turn_sharpness.size else 0.0
    semantic_drift_score = float(0.6 * step_mean + 0.4 * turn_mean)

    second_diff_norms = (
        np.linalg.norm(emb[2:] - 2 * emb[1:-1] + emb[:-2], axis=1) if n_segments >= 3 else np.array([], dtype=np.float32)
    )
    second_diff_mean = float(second_diff_norms.mean()) if second_diff_norms.size else 0.0

    skip2 = np.array([float(s_matrix[i, i + 2]) for i in range(n_segments - 2)], dtype=np.float32) if n_segments >= 3 else np.array([], dtype=np.float32)
    skip2_mean = float(skip2.mean()) if skip2.size else float("nan")

    smoothness_score = float((1.0 / (1.0 + second_diff_mean)) * 0.5 + (skip2_mean if np.isfinite(skip2_mean) else 0.0) * 0.5)

    s_wo_diag = s_matrix.copy().astype(np.float32)
    np.fill_diagonal(s_wo_diag, np.nan)
    centrality_all = np.nanmean(s_wo_diag, axis=1).astype(np.float32)

    c_mean = float(np.nanmean(centrality_all))
    c_std = float(np.nanstd(centrality_all))
    centrality_z = (centrality_all - c_mean) / c_std if c_std > 0 else np.zeros_like(centrality_all)
    outlierness = (-centrality_z).astype(np.float32)

    outlier_order = np.argsort(outlierness)[::-1][: min(int(cfg["top_n_outliers"]), n_segments)]
    segment_outliers: list[dict[str, Any]] = []
    for idx in outlier_order:
        idx = int(idx)
        segment_outliers.append(
            {
                "index": idx,
                "path": paths[idx],
                "role": str(roles[idx]),
                "centrality_all": float(centrality_all[idx]),
                "centrality_z": float(centrality_z[idx]),
                "outlierness": float(outlierness[idx]),
            }
        )

    intro_to_main_best = []
    intro_to_conc_best = []
    for i in intro_idx:
        i = int(i)
        if main_idx.size:
            intro_to_main_best.append(float(np.max(s_matrix[i, main_idx])))
        if conc_idx.size:
            intro_to_conc_best.append(float(np.max(s_matrix[i, conc_idx])))

    intro_to_main_best = np.array(intro_to_main_best, dtype=np.float32)
    intro_to_conc_best = np.array(intro_to_conc_best, dtype=np.float32)

    coverage_main = float(np.mean(intro_to_main_best >= cfg["coverage_threshold"])) if intro_to_main_best.size else float("nan")
    coverage_conclusion = float(np.mean(intro_to_conc_best >= cfg["coverage_threshold"])) if intro_to_conc_best.size else float("nan")
    coverage_parts = [x for x in [coverage_main, coverage_conclusion] if np.isfinite(x)]
    coverage_score = float(np.mean(coverage_parts)) if coverage_parts else float("nan")

    novelty_profile = []
    for i in range(1, n_segments):
        sim_prev = float(np.max(s_matrix[i, :i]))
        if sim_prev >= cfg["novelty_high_threshold"]:
            label = "redundant"
        elif sim_prev <= cfg["novelty_low_threshold"]:
            label = "jump"
        else:
            label = "balanced"
        novelty_profile.append(
            {
                "index": i,
                "path": paths[i],
                "max_similarity_to_previous": sim_prev,
                "label": label,
            }
        )

    redundant_share = float(np.mean([x["label"] == "redundant" for x in novelty_profile])) if novelty_profile else float("nan")
    jump_share = float(np.mean([x["label"] == "jump" for x in novelty_profile])) if novelty_profile else float("nan")

    section_boundaries = []
    for i in range(n_segments - 1):
        if parents[i] != parents[i + 1]:
            section_boundaries.append(i)

    boundary_transitions = []
    for i in section_boundaries:
        boundary_transitions.append(
            {
                "from_index": int(i),
                "to_index": int(i + 1),
                "from_parent": str(parents[i]),
                "to_parent": str(parents[i + 1]),
                "similarity": float(s_matrix[i, i + 1]),
                "z": float(adj_z[i]) if i < len(adj_z) else None,
                "from_path": paths[i],
                "to_path": paths[i + 1],
            }
        )

    problem_flags = []
    issues = []

    if np.isfinite(adj_mean) and adj_mean < 0.65:
        problem_flags.append(
            {"type": "low_local_coherence", "severity": "high", "metric": "adj_mean", "value": float(adj_mean), "threshold": 0.65}
        )

    if np.isfinite(coverage_score) and coverage_score < 0.50:
        problem_flags.append(
            {"type": "low_goal_coverage", "severity": "high", "metric": "coverage_score", "value": float(coverage_score), "threshold": 0.50}
        )

    if np.isfinite(conc_delta) and conc_delta < -0.05:
        problem_flags.append(
            {
                "type": "conclusion_main_misalignment",
                "severity": "medium",
                "metric": "conclusion_delta",
                "value": float(conc_delta),
                "threshold": -0.05,
            }
        )

    if np.isfinite(redund_90) and redund_90 > 0.25:
        problem_flags.append(
            {"type": "high_redundancy", "severity": "medium", "metric": "redund_90", "value": float(redund_90), "threshold": 0.25}
        )

    if np.isfinite(jump_share) and jump_share > 0.30:
        problem_flags.append(
            {"type": "high_semantic_jumps", "severity": "medium", "metric": "jump_share", "value": float(jump_share), "threshold": 0.30}
        )

    for tr in weakest_transitions:
        sev = "high" if tr["adj_z"] <= -1.5 else ("medium" if tr["adj_z"] <= -1.0 else "low")
        issues.append(
            {
                "type": "weak_transition",
                "severity": sev,
                "where": f"{tr['from_index']}->{tr['to_index']}",
                "from_path": tr["from_path"],
                "to_path": tr["to_path"],
                "metric": "adj_similarity",
                "value": float(tr["adj_similarity"]),
                "aux_z": float(tr["adj_z"]),
                "recommendation": "    /   .",
            }
        )

    for s in segment_outliers:
        if s["centrality_z"] <= cfg["outlier_z"]:
            sev = "high" if s["centrality_z"] <= -1.5 else "medium"
            issues.append(
                {
                    "type": "outlier_segment",
                    "severity": sev,
                    "where": f"index={s['index']}",
                    "from_path": s["path"],
                    "to_path": None,
                    "metric": "centrality_z",
                    "value": float(s["centrality_z"]),
                    "aux_z": float(s["outlierness"]),
                    "recommendation": "    /   .",
                }
            )

    for b in boundary_transitions:
        z = b["z"] if b["z"] is not None else 0.0
        if z <= -1.0:
            sev = "high" if z <= -1.5 else "medium"
            issues.append(
                {
                    "type": "weak_section_boundary",
                    "severity": sev,
                    "where": f"{b['from_index']}->{b['to_index']}",
                    "from_path": b["from_path"],
                    "to_path": b["to_path"],
                    "metric": "boundary_similarity",
                    "value": float(b["similarity"]),
                    "aux_z": float(z),
                    "recommendation": "    /   .",
                }
            )

    rank = {"high": 0, "medium": 1, "low": 2}
    issues = sorted(issues, key=lambda x: (rank.get(x["severity"], 9), x.get("value", 1.0)))
    issues = issues[: int(cfg["top_n_issues"])]

    risk_score = 0.0
    if np.isfinite(adj_mean):
        risk_score += max(0.0, (0.70 - adj_mean)) * 100.0
    if np.isfinite(semantic_drift_score):
        risk_score += min(20.0, semantic_drift_score * 15.0)
    if np.isfinite(smoothness_score):
        risk_score += max(0.0, (0.60 - smoothness_score)) * 35.0
    if np.isfinite(coverage_score):
        risk_score += max(0.0, (0.70 - coverage_score)) * 45.0
    if np.isfinite(redund_90):
        risk_score += max(0.0, redund_90 - 0.10) * 60.0
    if np.isfinite(jump_share):
        risk_score += jump_share * 30.0
    risk_score = float(min(100.0, max(0.0, risk_score)))

    report: dict[str, Any] = {
        "version": "v3",
        "model": "FRIDA",
        "n_segments": int(n_segments),
        "diagnostic_metrics": {
            "local_coherence": {
                "adj_mean": float(adj_mean),
                "adj_std": float(adj_std),
                "weak_transition_count": int(len(weak_transition_indices)),
                "weak_transition_share": float(len(weak_transition_indices) / adj.size) if adj.size else None,
                "weakest_transitions_top_n": weakest_transitions,
            },
            "structure_consistency": {
                "intro_to_main_mean": float(intro_to_main_mean) if np.isfinite(intro_to_main_mean) else None,
                "conclusion_to_main_mean": float(conc_to_main_mean) if np.isfinite(conc_to_main_mean) else None,
                "main_main_mean": float(main_main_mean) if np.isfinite(main_main_mean) else None,
                "intro_delta": float(intro_delta) if np.isfinite(intro_delta) else None,
                "conclusion_delta": float(conc_delta) if np.isfinite(conc_delta) else None,
            },
            "drift_and_smoothness": {
                "semantic_drift_score": float(semantic_drift_score),
                "step_mean_norm": float(step_mean),
                "turn_mean_sharpness": float(turn_mean),
                "second_diff_mean": float(second_diff_mean),
                "skip2_mean": float(skip2_mean) if np.isfinite(skip2_mean) else None,
                "smoothness_score": float(smoothness_score),
            },
            "centrality": {
                "mean": float(np.nanmean(centrality_all)),
                "std": float(np.nanstd(centrality_all)),
                "top_outliers": segment_outliers,
            },
            "coverage": {
                "threshold": float(cfg["coverage_threshold"]),
                "coverage_main": float(coverage_main) if np.isfinite(coverage_main) else None,
                "coverage_conclusion": float(coverage_conclusion) if np.isfinite(coverage_conclusion) else None,
                "coverage_score": float(coverage_score) if np.isfinite(coverage_score) else None,
            },
            "novelty_redundancy_profile": {
                "low_threshold": float(cfg["novelty_low_threshold"]),
                "high_threshold": float(cfg["novelty_high_threshold"]),
                "redundant_share": float(redundant_share) if np.isfinite(redundant_share) else None,
                "jump_share": float(jump_share) if np.isfinite(jump_share) else None,
                "items": novelty_profile,
            },
            "section_boundaries": boundary_transitions,
            "redundancy": {
                "threshold": float(cfg["redundancy_threshold"]),
                "redund_90": float(redund_90) if np.isfinite(redund_90) else None,
            },
        },
        "interpretation": {
            "global_flags": problem_flags,
            "localized_issues": issues,
        },
        "summary": {
            "risk_score_v3": float(risk_score),
            "note": "Analyze localized_issues first, then global_flags. risk_score_v3 is supportive.",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

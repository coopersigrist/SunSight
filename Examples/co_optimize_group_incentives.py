"""
Fast heuristic co-optimization of grouped federal incentives (ZIP → group, per-group $).

Optimizes a *scalarized* score: weighted min–max normalized (Carbon, Energy Potential,
Racial Equity, Income Equity) minus an optional penalty on NEAT proportion MAE.

Designed for the same ``work_df`` / ``INCENTIVE_GRID`` / ``objectives`` / ``zips_df``
pipeline as ``greedy_neat_match_grouping.ipynb``.

Runtime knobs (for <~1 hour on typical laptop):
  - ``incentive_grid``: pass a *coarser* list than the full greedy grid (e.g. step 1000).
  - ``max_iters``, ``assignment_passes``: fewer passes = faster.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _build_placements(
    df: pd.DataFrame,
    group_ids: np.ndarray,
    group_incentives: Sequence[float],
) -> Dict[int, float]:
    pred_panels: Dict[int, float] = {}
    for i, r in enumerate(df.itertuples(index=False)):
        g = int(group_ids[i])
        inc = group_incentives[g]
        pred_panels[int(r.zip)] = float(r.pred_panels_by_incentive[inc])
    return pred_panels


def evaluate_strategy(
    df: pd.DataFrame,
    group_ids: np.ndarray,
    group_incentives: Sequence[float],
    zips_df: pd.DataFrame,
    objectives: Sequence[Any],
) -> Dict[str, Any]:
    """Return placements, proportion MAE/RMSE/score, and objective dict (same names as objectives)."""
    pred_panels = _build_placements(df, group_ids, group_incentives)
    total_pred = float(sum(pred_panels.values()))
    if total_pred <= 0:
        return {
            "mae": 1.0,
            "rmse": 1.0,
            "match_score": 0.0,
            "placements": pred_panels,
            "objectives": {},
        }

    pred_prop = {z: p / total_pred for z, p in pred_panels.items()}
    keys = sorted(set(pred_prop.keys()) | set(df["zip"].tolist()))
    target_map = {int(r.zip): float(r.target_prop) for r in df.itertuples(index=False)}
    diffs = np.array([pred_prop.get(k, 0.0) - target_map.get(k, 0.0) for k in keys], dtype=float)
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs * diffs)))
    score = float(max(0.0, 1.0 - mae * len(keys) / 2.0))

    obj_vals: Dict[str, float] = {}
    for obj in objectives:
        obj_vals[obj.name] = float(obj.calc(zips_df, pred_panels))

    return {
        "mae": mae,
        "rmse": rmse,
        "match_score": score,
        "placements": pred_panels,
        "objectives": obj_vals,
    }


def metric_envelope_uniform_incentive(
    df: pd.DataFrame,
    incentive_grid: Sequence[int],
    zips_df: pd.DataFrame,
    objectives: Sequence[Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Min/max of each objective when every ZIP gets the same incentive (one value from grid).
    Used only for cheap min–max scaling of the scalarized score.
    """
    lows = {obj.name: float("inf") for obj in objectives}
    highs = {obj.name: float("-inf") for obj in objectives}
    for inc in incentive_grid:
        gids = np.zeros(len(df), dtype=int)
        ginc = [float(inc)]
        ev = evaluate_strategy(df, gids, ginc, zips_df, objectives)
        for k, v in ev["objectives"].items():
            if not np.isfinite(v):
                continue
            lows[k] = min(lows[k], v)
            highs[k] = max(highs[k], v)
    for obj in objectives:
        if not np.isfinite(lows[obj.name]):
            lows[obj.name] = 0.0
        if not np.isfinite(highs[obj.name]):
            highs[obj.name] = 1.0
        if highs[obj.name] <= lows[obj.name]:
            highs[obj.name] = lows[obj.name] + 1e-9
    return lows, highs


def scalarized_loss(
    objectives: Mapping[str, float],
    mae: float,
    lo: Mapping[str, float],
    hi: Mapping[str, float],
    weights: Mapping[str, float],
    alpha_prop: float,
) -> float:
    """
    Lower is better. Maximizes weighted normalized objectives; adds alpha_prop * MAE (proportion mismatch).
    """
    acc = 0.0
    for k, w in weights.items():
        if w == 0:
            continue
        v = float(objectives.get(k, 0.0))
        t = (v - float(lo[k])) / (float(hi[k]) - float(lo[k]) + 1e-12)
        t = float(np.clip(t, 0.0, 1.0))
        acc += w * t
    return float(alpha_prop * mae - acc)


def initial_group_ids_target_quantiles(df: pd.DataFrame, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    rank = pd.qcut(df["target_prop"].rank(method="first"), q=n_groups, labels=False, duplicates="drop")
    gids = rank.to_numpy(dtype=int)
    if int(gids.max()) + 1 < n_groups:
        gids = rng.integers(0, n_groups, size=len(df), endpoint=False)
    return gids


def initial_incentives_spaced(incentive_grid: Sequence[int], n_groups: int) -> List[int]:
    grid = list(incentive_grid)
    if not grid:
        raise ValueError("incentive_grid must be non-empty")
    idx = np.linspace(0, len(grid) - 1, n_groups).round().astype(int)
    return [grid[i] for i in idx]


def co_optimize_group_incentives(
    df: pd.DataFrame,
    n_groups: int,
    zips_df: pd.DataFrame,
    objectives: Sequence[Any],
    incentive_grid: Sequence[int],
    *,
    weights: Optional[Mapping[str, float]] = None,
    alpha_prop: float = 0.08,
    max_iters: int = 8,
    assignment_passes: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    Coordinate descent on group incentives + a few proportion-MAE ZIP assignment passes.

    Parameters
    ----------
    weights : dict keyed by objective names, e.g.
        {"Carbon Offset": 0.25, "Energy Potential": 0.25, "Racial Equity": 0.25, "Income Equity": 0.25}
    alpha_prop : trade-off for NEAT proportion MAE (larger = prioritize matching NEAT shares).
    assignment_passes : number of outer iterations that run MAE-based ZIP reassignment before
        switching to scalarized incentive updates only (0 = never; 2 = default warm-start).
    """
    rng = rng or np.random.default_rng(0)
    if weights is None:
        weights = {
            "Carbon Offset": 0.25,
            "Energy Potential": 0.25,
            "Racial Equity": 0.25,
            "Income Equity": 0.25,
        }

    lo, hi = metric_envelope_uniform_incentive(df, incentive_grid, zips_df, objectives)
    group_ids = initial_group_ids_target_quantiles(df, n_groups, rng)
    group_incentives = initial_incentives_spaced(incentive_grid, n_groups)

    ev_cur = evaluate_strategy(df, group_ids, group_incentives, zips_df, objectives)
    best_loss = scalarized_loss(ev_cur["objectives"], ev_cur["mae"], lo, hi, weights, alpha_prop)
    best_group_ids = group_ids.copy()
    best_group_incentives = list(group_incentives)
    best_ev = ev_cur

    for it in range(max_iters):
        changed = False

        if it < assignment_passes:
            current_eval = evaluate_strategy(df, group_ids, group_incentives, zips_df, objectives)
            denom = max(float(sum(current_eval["placements"].values())), 1e-9)
            for i, r in enumerate(df.itertuples(index=False)):
                cur_g = int(group_ids[i])
                cur_best = cur_g
                cur_best_loss = None
                for g in range(n_groups):
                    inc = group_incentives[g]
                    local_pred = float(r.pred_panels_by_incentive[inc])
                    local_pred_prop = local_pred / denom
                    loss = abs(local_pred_prop - float(r.target_prop))
                    if cur_best_loss is None or loss < cur_best_loss:
                        cur_best_loss = loss
                        cur_best = g
                if cur_best != cur_g:
                    group_ids[i] = cur_best
                    changed = True

        for g in range(n_groups):
            best_inc = group_incentives[g]
            best_l = None
            for inc in incentive_grid:
                trial = list(group_incentives)
                trial[g] = inc
                ev = evaluate_strategy(df, group_ids, trial, zips_df, objectives)
                L = scalarized_loss(ev["objectives"], ev["mae"], lo, hi, weights, alpha_prop)
                if best_l is None or L < best_l:
                    best_l = L
                    best_inc = inc
            if best_inc != group_incentives[g]:
                group_incentives[g] = best_inc
                changed = True

        ev_now = evaluate_strategy(df, group_ids, group_incentives, zips_df, objectives)
        L_now = scalarized_loss(ev_now["objectives"], ev_now["mae"], lo, hi, weights, alpha_prop)
        if L_now < best_loss:
            best_loss = L_now
            best_group_ids = group_ids.copy()
            best_group_incentives = list(group_incentives)
            best_ev = ev_now

        if not changed:
            break

    return best_group_ids, best_group_incentives, best_ev


def pareto_weight_sketches(
    df: pd.DataFrame,
    n_groups: int,
    zips_df: pd.DataFrame,
    objectives: Sequence[Any],
    incentive_grid: Sequence[int],
    weight_vectors: Sequence[Mapping[str, float]],
    **kwargs: Any,
) -> pd.DataFrame:
    """Run several scalarized optimizations (different weights); returns a summary table."""
    rows = []
    for wi, w in enumerate(weight_vectors):
        gids, ginc, ev = co_optimize_group_incentives(
            df, n_groups, zips_df, objectives, incentive_grid, weights=dict(w), **kwargs
        )
        row = {
            "sketch_id": wi,
            "weights": str(dict(w)),
            "mae": ev["mae"],
            "match_score": ev["match_score"],
        }
        row.update(ev["objectives"])
        rows.append(row)
    return pd.DataFrame(rows)

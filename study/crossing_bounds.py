"""Query-specific precision certificates for a fixed continuous transport budget.

These are consequences of the fractional-knapsack projection, not a different
uncertainty set or a new calibration rule. Costs/capacities are already extracted
from known geometry and the coherent predictor. No fine test outcome is an input.
"""
import numpy as np


def side_certificate(capacity, cost, budget):
    """Bound unfilled crossing mass without sorting.

    Infinite costs correspond to absent destinations and are omitted. Zero-cost
    crossings are filled for free. For positive rows let B=sum(p_i*delta_i),
    a=sum(p_i), d_min/min(delta_i), d_max/max(delta_i). The unfilled mass lies
    between (B-budget)_+/d_max and min(a,(B-budget)_+/d_min). Empty positive
    support has zero deficit. Arrays are nonnegative; the budget is nonnegative.
    """
    p, d = np.asarray(capacity, float), np.asarray(cost, float)
    if p.shape != d.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid capacities")
    if np.isnan(d).any() or (d < 0).any() or np.isnan(budget) or budget < 0:
        raise ValueError("Invalid costs or budget")
    allowed = np.isfinite(d)
    positive = allowed & (d > 0) & (p > 0)
    total_crossing = float(p[allowed].sum())
    if not positive.any():
        return {"crossing_mass": total_crossing, "positive_mass": 0., "saturation_budget": 0.,
                "minimum_positive_cost": 0., "maximum_positive_cost": 0.,
                "deficit_lower": 0., "deficit_upper": 0.}
    a = float(p[positive].sum())
    b = float(p[positive] @ d[positive])
    low, high = float(d[positive].min()), float(d[positive].max())
    shortfall = max(0., b-budget)
    return {"crossing_mass": total_crossing, "positive_mass": a, "saturation_budget": b,
            "minimum_positive_cost": low, "maximum_positive_cost": high,
            "deficit_lower": shortfall/high, "deficit_upper": min(a, shortfall/low)}


def query_certificate(geo, query, budget):
    """Extract both crossing sides in O(E) and bound coarse-width reduction."""
    q = np.asarray(query)
    if q.shape != geo["p"].shape or not np.isin(q, [0, 1]).all():
        raise ValueError("Expected a binary query in candidate order")
    sides = []
    for losing, gaining in [(0, 1), (1, 0)]:
        selected = (q[geo["u"]] == losing) & (q[geo["v"]] == gaining)
        delta = np.full(len(q), np.inf)
        np.minimum.at(delta, geo["u"][selected], geo["cost"][selected])
        sides.append(side_certificate(geo["p"], delta, budget))
    width = sum(s["crossing_mass"] for s in sides)
    lower = sum(s["deficit_lower"] for s in sides)
    upper = sum(s["deficit_upper"] for s in sides)
    return {"coarse_width": width, "narrowing_lower": lower, "narrowing_upper": upper,
            "saturation_budget": max(s["saturation_budget"] for s in sides),
            "plus": sides[0], "minus": sides[1]}

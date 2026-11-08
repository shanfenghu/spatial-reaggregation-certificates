"""Residual-mass certificates from a proportional feasible crossing plan."""
import crossing_bounds as legacy


def _replace_upper(side, budget):
    """Use H <= A(B-tau)_+/B, with zero deficit when B is zero.

    A is positive finite-cost mass and B its full crossing cost. Free rows have
    already crossed. This needs no ordering, outcomes, or extra transport solve.
    """
    result = dict(side)
    a, b = side["positive_mass"], side["saturation_budget"]
    upper = a * (max(0., b - budget) / b) if b > 0 else 0.
    result["legacy_deficit_upper"] = side["deficit_upper"]
    result["deficit_upper"] = upper
    return result


def side_certificate(capacity, cost, budget):
    """Return bounds for nonnegative capacities/costs and budget >= 0.

    Infinite costs mean absent destinations. The cost-range routine validates the
    inputs and supplies the unchanged lower bound and exact saturation cost.
    For B>0, uniformly crossing fraction min(1,tau/B) proves the upper bound.
    """
    return _replace_upper(legacy.side_certificate(capacity, cost, budget), budget)


def query_certificate(geo, query, budget):
    """Bound binary-query narrowing with O(E) extraction and no sorting.

    Both certificates use the same predictor, costs, query and budget.
    Cost-range values permit a per-query dominance check.
    """
    result = legacy.query_certificate(geo, query, budget)
    result["legacy_narrowing_upper"] = result["narrowing_upper"]
    for sign in ["plus", "minus"]:
        result[sign] = _replace_upper(result[sign], budget)
    result["narrowing_upper"] = sum(result[s]["deficit_upper"] for s in ["plus", "minus"])
    return result

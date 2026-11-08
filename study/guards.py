"""Validate binary query inputs and the coupling used as an endpoint witness."""
import numpy as np


def checked_query(geo, query, budget):
    """Reject fractional indicators before any integer conversion can lose them."""
    q = np.asarray(query)
    if q.shape != geo['p'].shape or not np.isin(q, [0,1]).all():
        raise ValueError('Expected a binary query in candidate order')
    if not np.isscalar(budget) or np.isnan(budget) or budget < 0:
        raise ValueError('Expected a nonnegative budget')
    return q.astype(float, copy=False)


def check_witness(geo, query, budget, coupling, expected, tolerance=1e-7):
    """Check finite nonnegative flow, row conservation, budget and objective.

    Column masses are free: only row masses and within-source arcs are fixed.
    This checks a numerical witness; it does not replace optimality reasoning.
    """
    q = checked_query(geo, query, budget)
    flow = np.asarray(coupling, dtype=float)
    if flow.shape != geo['cost'].shape or not np.isfinite(flow).all() or not np.isfinite(expected):
        raise ValueError('Invalid endpoint witness shape or nonfinite value')
    row_error = float(np.max(np.abs(geo['row'] @ flow - geo['p'])))
    value = float(q[geo['v']] @ flow)
    if (float(flow.min()) < -tolerance or row_error > tolerance or
            float(geo['cost'] @ flow) > budget+tolerance or abs(value-expected) > tolerance):
        raise ValueError('Endpoint witness violates a constraint or objective')
    return value

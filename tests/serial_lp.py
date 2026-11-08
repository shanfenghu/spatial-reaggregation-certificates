"""Keep oracle solves serial, consistent with the evaluated HiGHS configuration.

HiGHS shares a process scheduler: an earlier automatic-thread oracle solve can
prevent a later explicit one-thread solve. This changes no oracle formulation.
"""
import warnings
from scipy.optimize import linprog as scipy_linprog, OptimizeWarning


def linprog(*args, **kwargs):
    options = dict(kwargs.pop('options', {}), threads=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected', category=OptimizeWarning)
        return scipy_linprog(*args, options=options, **kwargs)

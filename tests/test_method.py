"""Independent LP oracle and degenerate-crossing controls for the exported scan."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from serial_lp import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"study"))
from engine import geometry, endpoint_scan
from compare import self_check


class ProjectionTests(unittest.TestCase):
    def test_scan_matches_independent_lp(self):
        rng = np.random.default_rng(7812)
        # A source has colocated opposite-side points; another has no crossing.
        frame = pd.DataFrame({
            "source": ["a"]*3+["b"]*2, "x": [0., 0., 2., 4., 5.],
            "y": [0.]*5, "truth": [0., 3., 7., 4., 6.],
            "weight": [1., 2., 3., 1., 0.]})
        geo = geometry(frame)
        for q in [np.zeros(5), np.ones(5), np.array([0, 1, 0, 1, 1])]+[
                rng.integers(0, 2, 5) for _ in range(15)]:
            for budget in [0., .017, .2, 2.]:
                for upper in [False, True]:
                    objective = q[geo["v"]]*(-1 if upper else 1)
                    fit = linprog(objective, A_eq=geo["row"], b_eq=geo["p"],
                                  A_ub=geo["cost"][None, :], b_ub=[budget],
                                  bounds=(0, None), method="highs")
                    self.assertTrue(fit.success)
                    expected = fit.fun*geo["total"]*(-1 if upper else 1)
                    self.assertAlmostEqual(endpoint_scan(geo, q, budget, upper)[0], expected, places=8)

    def test_documented_new_query_example(self):
        frame = pd.DataFrame({"source": ["a", "a", "b", "b"],
                              "x": [0., 1., 0., 1.], "y": [1., 1., 0., 0.],
                              "truth": [40., 60., 20., 0.], "weight": [1.]*4})
        geo = geometry(frame)
        q = np.array([1, 0, 0, 0])
        self.assertAlmostEqual(endpoint_scan(geo, q, 1/6, False)[0], 30.)
        self.assertAlmostEqual(endpoint_scan(geo, q, 1/6, True)[0], 70.)

    def test_rank_conic_box_and_hidden_outcome_controls(self):
        self_check()


if __name__ == "__main__":
    unittest.main()

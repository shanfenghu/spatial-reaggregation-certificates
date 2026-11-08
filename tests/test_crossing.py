"""Independent LP checks of the precision certificate and its boundary cases."""
import unittest
import numpy as np
from serial_lp import linprog
from crossing_bounds import side_certificate


class BudgetCertificateTests(unittest.TestCase):
    def test_independent_lp_random_and_edge_cases(self):
        rng = np.random.default_rng(20261104)
        cases = [(np.array([.1,.2,.7]), np.array([0.,1.,np.inf])),
                 (np.array([0.,1.]), np.array([.1,0.])),
                 (np.array([.4,.6]), np.array([.25,.25]))]
        for _ in range(80):
            p = rng.dirichlet(np.ones(9))
            d = rng.uniform(.001, 1, len(p))
            d[0], d[1] = 0., np.inf
            cases.append((p,d))
        for p,d in cases:
            finite = np.isfinite(d)
            total = float(p[finite] @ d[finite])
            for budget in [0., total*.1, total*.5, total, total+1.]:
                result = side_certificate(p,d,budget)
                fit = linprog(-np.ones(finite.sum()), A_ub=d[finite][None,:],
                              b_ub=[budget], bounds=list(zip(np.zeros(finite.sum()),p[finite])),
                              method="highs")
                self.assertTrue(fit.success)
                deficit = float(p[finite].sum()+fit.fun)
                self.assertGreaterEqual(deficit+1e-10,result["deficit_lower"])
                self.assertLessEqual(deficit-1e-10,result["deficit_upper"])
                if len(set(d[finite & (d > 0) & (p > 0)])) <= 1:
                    self.assertAlmostEqual(result["deficit_lower"],deficit,places=10)
                    self.assertAlmostEqual(result["deficit_upper"],deficit,places=10)

    def test_infinite_budget_and_empty_positive_rows(self):
        self.assertEqual(side_certificate([.4,.6],[0.,np.inf],0.)["deficit_upper"],0.)
        self.assertEqual(side_certificate([.4,.6],[.1,.3],np.inf)["deficit_upper"],0.)


if __name__ == "__main__":
    unittest.main()

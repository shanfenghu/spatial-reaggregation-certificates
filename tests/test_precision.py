"""Independent LP checks of the replacement certificate, not empirical evidence."""
import unittest
import numpy as np
from serial_lp import linprog
from precision import side_certificate


class UniformPrecisionTests(unittest.TestCase):
    def test_bounds_and_dominance_against_415_lp_optima(self):
        rng = np.random.default_rng(20261105)
        cases = [(np.array([.1, .2, .7]), np.array([0., 1., np.inf])),
                 (np.array([0., 1.]), np.array([.1, 0.])),
                 (np.array([.4, .6]), np.array([.25, .25]))]
        for _ in range(80):
            p = rng.dirichlet(np.ones(9))
            d = rng.uniform(.001, 1, len(p))
            d[0], d[1] = 0., np.inf
            cases.append((p, d))
        for p, d in cases:
            finite = np.isfinite(d)
            b = float(p[finite] @ d[finite])
            for tau in [0., .1*b, .5*b, b, b+1.]:
                cert = side_certificate(p, d, tau)
                fit = linprog(-np.ones(finite.sum()), A_ub=d[finite][None, :],
                              b_ub=[tau], bounds=list(zip(np.zeros(finite.sum()), p[finite])),
                              method="highs")
                self.assertTrue(fit.success)
                h = float(p[finite].sum()+fit.fun)
                self.assertLessEqual(cert["deficit_lower"], h+1e-10)
                self.assertGreaterEqual(cert["deficit_upper"], h-1e-10)
                self.assertLessEqual(cert["deficit_upper"], cert["legacy_deficit_upper"]+1e-10)
                if len(set(d[finite & (d > 0) & (p > 0)])) <= 1:
                    self.assertAlmostEqual(cert["deficit_upper"], h, places=10)
                    self.assertAlmostEqual(cert["deficit_lower"], h, places=10)

    def test_strict_tightening_and_uniform_witness(self):
        p, d, tau = np.array([.5, .5]), np.array([.1, 1.]), .5
        cert = side_certificate(p, d, tau)
        z = p*tau/float(p@d)
        self.assertAlmostEqual(float(z@d), tau)
        self.assertAlmostEqual(float((p-z).sum()), cert["deficit_upper"])
        self.assertLess(cert["deficit_upper"], cert["legacy_deficit_upper"])

    def test_empty_free_absent_and_infinite_budget(self):
        for p, d, tau in [([], [], 0.), ([.4, .6], [0., np.inf], 0.),
                          ([.4, .6], [np.inf, np.inf], 0.),
                          ([.4, .6], [.1, .3], np.inf)]:
            cert = side_certificate(p, d, tau)
            self.assertEqual(cert["deficit_upper"], 0.)
            self.assertEqual(cert["deficit_lower"], 0.)


if __name__ == "__main__":
    unittest.main()

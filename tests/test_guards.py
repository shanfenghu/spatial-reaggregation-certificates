"""Regression tests for fractional indicators and invalid endpoint witnesses."""
import unittest
import numpy as np
import pandas as pd
from engine import geometry, endpoint_scan, endpoint_lp
from guards import check_witness


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.geo=geometry(pd.DataFrame(dict(source=['a','a'],x=[0.,1.],y=[0.,0.],
            truth=[1.,1.],weight=[1.,1.])))

    def test_fractional_indicator_rejected_without_truncation(self):
        for q in [[.5,0.],[1.,.2],[np.nan,0.],[np.inf,0.],[1.,0.,0.]]:
            for method in [endpoint_scan,endpoint_lp]:
                with self.subTest(query=q,method=method.__name__),self.assertRaises(ValueError):
                    method(self.geo,np.array(q),.1,True)

    def test_binary_lists_and_budget_edges(self):
        for budget in [0.,.1,np.inf]:
            for upper in [False,True]:
                self.assertAlmostEqual(endpoint_scan(self.geo,[1,0],budget,upper)[0],
                                       endpoint_lp(self.geo,[1,0],budget,upper)[0])
        for budget in [-.1,np.nan]:
            with self.assertRaises(ValueError):endpoint_scan(self.geo,[1,0],budget,True)

    def test_witness_is_checked_without_rounding(self):
        # Row-major within-source arcs: the identity coupling is feasible.
        valid=np.array([.5,0.,0.,.5])
        self.assertEqual(check_witness(self.geo,[1,0],0.,valid,.5),.5)
        for flow,value in [(np.array([.5,-.1,.1,.5]),.6),
                           (valid*.5,.25),(np.array([np.nan,0.,0.,.5]),.5),
                           (valid,.4),(np.array([0.,.5,.5,0.]),.5)]:
            with self.subTest(flow=flow,value=value),self.assertRaises(ValueError):
                check_witness(self.geo,[1,0],0.,flow,value)


if __name__=='__main__':unittest.main()

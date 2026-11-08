"""Geometry and permission controls for the changed-radius workload."""
import json
import unittest
import numpy as np
import pandas as pd
from extension import query_union, endpoints
import compare as study


class QueryExtensionTests(unittest.TestCase):
    def test_geometry_deduplication_and_outcome_independence(self):
        # Two supplied centers coincide; half-radius duplicates the original,
        # while double radius adds a crossing. Fine redistributions keep totals.
        frame = pd.DataFrame({"lat": [0., 0., 0.], "lon": [0., .04, .2],
                              "source": ["a"] * 3, "truth": [1., 2., 3.]})
        mask = np.packbits([1, 0, 0], bitorder="little").tobytes().hex()
        registered = pd.DataFrame({"mask_hex": [mask], "all_center_ids": [json.dumps(["1", "2"])]})
        centers = pd.DataFrame({"ORISPL": ["1", "2"], "LAT": [0., 0.], "LON": [0., 0.]})
        a, controls = query_union(frame, registered, centers, [1.5, 6.])
        frame["truth"] = [5., 0., 1.]
        b, _ = query_union(frame, registered, centers, [1.5, 6.])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)
        self.assertEqual(controls["duplicate_added_pairs"], 1)
        self.assertEqual(controls["original_mask_pairs"], 2)

    def test_expanded_permission_changes_only_declared_catalog_radius(self):
        frame = pd.DataFrame({"source": ["a", "a"], "x": [0., 1.], "y": [0., 0.],
                              "truth": [1., 1.], "weight": [1., 1.]})
        c = study.case(frame, [np.array([1, 0])])
        fit = {"radii": {"aggregate_global": .1}}
        a = endpoints(c, "aggregate_global", fit, {"aggregate_global": .3}, None, 1.)
        b = endpoints(c, "aggregate_global_expanded", fit, {"aggregate_global": .3}, None, 1.)
        self.assertTrue(np.allclose(a[:2], [[.4], [.6]]))
        self.assertTrue(np.allclose(b[:2], [[.2], [.8]]))


if __name__ == "__main__":
    unittest.main()

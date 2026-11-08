"""Controls for paired map selection and invariant equal-map weighting."""
import unittest
import numpy as np
import pandas as pd
from composition import summarize_pair


class CompositionTests(unittest.TestCase):
    def test_selection_can_reverse_unmatched_means_without_paired_change(self):
        original = pd.DataFrame({"transport_scan": [0., .9, .8], "baseline": [.5, .7, .6]}, index=["a", "b", "c"])
        added = original.loc[["c", "b"]].copy()
        pairs, s = summarize_pair(original, added, "baseline")
        self.assertAlmostEqual(s["full_original_contrast"], -1. / 30.)
        self.assertAlmostEqual(s["selection_component"], 7. / 30.)
        self.assertAlmostEqual(s["within_map_component"], 0.)
        self.assertTrue(np.allclose(pairs.paired_change, 0.))

    def test_map_alignment_and_paired_change_are_not_query_count_weighting(self):
        original = pd.DataFrame({"transport_scan": [.2, .3], "baseline": [.4, .5]}, index=["a", "b"])
        added = pd.DataFrame({"transport_scan": [.8, .6], "baseline": [.5, .5]}, index=["b", "a"])
        pairs, s = summarize_pair(original, added, "baseline")
        self.assertAlmostEqual(s["original_contrast"], -.2)
        self.assertAlmostEqual(s["added_contrast"], .2)
        self.assertAlmostEqual(s["paired_change"], .4)
        self.assertEqual(s["narrower_to_wider"], 2)
        self.assertAlmostEqual(s["paired_change_leave_one_out_min"], .3)
        self.assertAlmostEqual(s["paired_change_leave_one_out_max"], .5)


if __name__ == "__main__":
    unittest.main()

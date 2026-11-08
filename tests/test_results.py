"""Scientific reproduction accepts numerical noise but rejects changed findings."""
from pathlib import Path
import tempfile
import unittest

from reproduce import compare_summary


class SummaryChecks(unittest.TestCase):
    def check_pair(self, expected, actual, passes):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'reference.csv', Path(temp)/'actual.csv'
            a.write_text(expected)
            b.write_text(actual)
            if passes:
                self.assertEqual(compare_summary(a, b)['status'], 'passed')
            else:
                with self.assertRaises(AssertionError):
                    compare_summary(a, b)

    def test_platform_noise_and_near_zero_pass(self):
        self.check_pair('method,queries,width,offset\nx,7,0.7942,0.0\n',
                        'method,queries,width,offset\nx,7,0.794200003,0.0000000004\n', True)

    def test_substantive_drift_fails(self):
        self.check_pair('width\n0.7942\n', 'width\n0.7943\n', False)

    def test_discrete_decisions_and_labels_remain_exact(self):
        for a, b in [('queries\n100000000\n', 'queries\n100000001\n'),
                     ('queries\n7.0\n', 'queries\n7.000000001\n'),
                     ('method\nx\n', 'method\ny\n')]:
            with self.subTest(a=a):
                self.check_pair(a, b, False)

    def test_schema_rows_and_missingness_remain_exact(self):
        for a, b in [('x,y\n1,2\n', 'y,x\n1,2\n'),
                     ('x\n1\n', 'x\n1\n2\n'),
                     ('x,y\n1,2\n', 'x,y\n1\n'),
                     ('x,y\n1,\n', 'x,y\n1,0.0\n'),
                     ('x\nnan\n', 'x\n0.0\n')]:
            with self.subTest(a=a):
                self.check_pair(a, b, False)

    def test_nonfinite_signs_and_positions(self):
        self.check_pair('x,y\nnan,inf\n', 'x,y\nNaN,inf\n', True)
        self.check_pair('x\ninf\n', 'x\n-inf\n', False)


if __name__ == '__main__':
    unittest.main()

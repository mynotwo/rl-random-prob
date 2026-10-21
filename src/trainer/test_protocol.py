import unittest

from protocol import pass_at_k, split_rows


class ProtocolTest(unittest.TestCase):
    def test_split_is_independent_of_training_seed(self):
        rows = [{"x": i} for i in range(10)]
        first = split_rows(rows, 0.8, 777)
        second = split_rows(rows, 0.8, 777)
        self.assertEqual(first[2:], second[2:])
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)

    def test_pass_at_k_boundaries(self):
        self.assertEqual(pass_at_k(0, 32, 8), 0.0)
        self.assertEqual(pass_at_k(32, 32, 8), 1.0)
        self.assertAlmostEqual(pass_at_k(1, 32, 1), 1 / 32)


if __name__ == "__main__":
    unittest.main()

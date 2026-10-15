import unittest
from unittest.mock import patch

import numpy as np

from bestrand import estimate, random_select, select


class MethodsTest(unittest.TestCase):
    def test_binary_and_continuous_candidates(self):
        for scores in [np.arange(45).reshape(3, 15) / 44, np.arange(45).reshape(3, 15) % 2]:
            with self.subTest(binary=np.isin(scores, (0, 1)).all()):
                indices = select(scores, 3, seed=7)
                np.testing.assert_array_equal(indices, select(scores, 3, seed=7))
                self.assertEqual(len(set(indices)), 3)
                self.assertTrue(np.all((0 <= indices) & (indices < 15)))

    def test_select_minimizes_pool_imbalance_and_breaks_ties(self):
        class Draws:
            def __init__(self, draws):
                self.draws = iter(draws)

            def choice(self, *args, **kwargs):
                return np.asarray(next(self.draws))

        # Pool mean 0.5: [0, 3] and [1, 2] both match perfectly;
        # the sorted original-index tuple [0, 3] wins, keeping draw order [3, 0].
        with patch("bestrand.methods._rng", side_effect=[Draws([[0, 1, 2, 3]]),
                    Draws([[0, 2], [1, 2], [3, 0]])]):
            result = select([[0, 0.25, 0.75, 1]], 2, candidate_count=3)
        np.testing.assert_array_equal(result, [3, 0])

    def test_pool_reference_ignores_unselected_columns(self):
        class PoolDraw:
            def choice(self, *args, **kwargs):
                return np.asarray([0, 2, 4, 6])

        from bestrand.methods import _rng
        def rng(seed, stream):
            return PoolDraw() if stream == "pool" else _rng(seed, stream)
        first = np.array([[0, 0.1, 1, 0.1, 0, 0.1, 1, 0.1]])
        second = first.copy()
        second[:, [1, 3, 5, 7]] = 1000
        with patch("bestrand.methods._rng", side_effect=rng):
            np.testing.assert_array_equal(select(first, 2, pool_multiplier=2),
                                          select(second, 2, pool_multiplier=2))

    def test_full_budget_and_saturated_pool(self):
        scores = np.arange(20).reshape(2, 10) / 19
        self.assertEqual(set(select(scores, 10)), set(range(10)))
        np.testing.assert_allclose(estimate(scores[:, select(scores, 10)]), scores.mean(axis=1), atol=1e-15)
        self.assertEqual(len(set(select(scores, 7))), 7)
        self.assertEqual(set(select(scores, 4, pool_multiplier=1)),
                         set(select(scores, 4, pool_multiplier=1, candidate_count=1)))

    def test_random_and_estimator(self):
        indices = random_select(30, 8, seed=19)
        np.testing.assert_array_equal(indices, random_select(30, 8, seed=19))
        self.assertEqual(len(set(indices)), 8)
        self.assertTrue(np.all(np.diff(indices) > 0))
        np.testing.assert_allclose(estimate([[0, 1, 1], [0.1, 0.4, 0.7]]), [2 / 3, 0.4])

    def test_invalid_inputs(self):
        for scores in [[], [[np.nan]], [[np.inf]], [1, 2], np.zeros((0, 3))]:
            with self.assertRaises(ValueError):
                select(scores, 1)
            with self.assertRaises(ValueError):
                estimate(scores)
        for kwargs in [{"k": 0}, {"k": 3}, {"k": 1.5}, {"k": True}, {"seed": -1},
                       {"seed": 0.5}, {"pool_multiplier": 0}, {"candidate_count": -2}]:
            arguments = {"k": 1, "seed": 0, **kwargs}
            with self.assertRaises(ValueError):
                select([[0, 1]], **arguments)
        with self.assertRaises(ValueError):
            random_select(3, 4)


if __name__ == "__main__":
    unittest.main()

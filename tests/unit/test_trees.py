from unittest import TestCase

import numpy as np

from fixed_income import trees


class TestTrees(TestCase):
    def test_initialize_two_period_tree(self):
        actual = trees.initialize_tree(maturity=1, time_step=0.5, is_zero=False)
        expected = np.array([[0, 0], [0, 0]])
        self.assertEquals(actual.tolist(), expected.tolist())

    def test_initialize_two_period_zero_tree(self):
        tree = trees.initialize_tree(maturity=1, time_step=0.5, is_zero=True)
        expected = (3, 3, 2)
        self.assertEquals(tree.shape, expected)

    def test_ho_lee(self):
        r0 = 0.050682155
        dt = 0.5

        tree = trees.initialize_tree(maturity=1, time_step=dt, is_zero=False)
        tree[0, 0] = r0

        actual_rate_tree, actual_zero_tree = trees.ho_lee(theta=-0.072299819, rate_tree=tree, period=1,
                                                          sigma=0.00671631656750658, time_step=dt)

        expected_rate_tree = [[0.05068215, 0.0192814], [0.0, 0.00978309]]
        expected_zero_tree = [[0.96792141, 0.99040562, 1.0], [0.0, 0.9951204, 1.0], [0.0, 0.0, 1.0]]
        self.assertTrue(np.isclose(actual_rate_tree, expected_rate_tree).all())
        self.assertTrue(np.isclose(actual_zero_tree, expected_zero_tree).all())

    def test_black_derman_toy(self):
        r0 = 0.050682155
        dt = 0.5

        tree = trees.initialize_tree(maturity=1, time_step=dt, is_zero=False)
        tree[0, 0] = r0

        actual_rate_tree, actual_zero_tree = trees.black_derman_toy(theta=-0.0969608599011977, rate_tree=tree, period=1,
                                                                    sigma=0.22225391407878499, time_step=dt)

        expected_rate_tree = [[0.05068215, 0.05650057],
                              [0.0, 0.04126176]]
        expected_zero_tree = [[0.95144404, 0.97214502,  1.0],
                              [0., 0.97958048, 1.0],
                              [0., 0., 1.0]]
        self.assertTrue(np.isclose(actual_rate_tree, expected_rate_tree).all())
        self.assertTrue(np.isclose(actual_zero_tree, expected_zero_tree).all())

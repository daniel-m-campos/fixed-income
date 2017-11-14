import unittest

import numpy as np
from fixed_income import mortgages


class TestFunctions(unittest.TestCase):
    def test_coupon(self):
        loan = 100000
        maturity = 26
        mortgage_rate = 0.04492
        freq = 4
        actual = mortgages.coupon(loan, maturity, mortgage_rate, freq)

        dt = 1 / freq
        periods = np.arange(1, int(maturity / dt) + 1)

        discounts = (1 + mortgage_rate * dt) ** -periods

        expected = loan / np.sum(discounts)
        self.assertAlmostEqual(actual, expected)

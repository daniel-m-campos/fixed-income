import unittest

import numpy as np

from fixed_income import futures


class TestConversionFactor(unittest.TestCase):
    def test_2_year(self):
        years = 1
        months = 11
        days = 29
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.906258
        actual = futures.conversion_factor('ZT', coupon=0.75 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_3_year(self):
        years = 3
        months = 0
        days = 14
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.867956
        actual = futures.conversion_factor('Z3N', coupon=1.125 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_5_year(self):
        years = 4
        months = 11
        days = 29
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.837079
        actual = futures.conversion_factor('ZF', coupon=2.125 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_10_year(self):
        years = 9
        months = 5
        days = 14
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.815653
        actual = futures.conversion_factor('ZN', coupon=3.375 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_classic_bonds(self):
        years = 20
        months = 11
        days = 13
        time_to_maturity = years + months / 12 + days / 365

        expected = 1.044053
        actual = futures.conversion_factor('ZB', coupon=6.375 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_ultra_bonds(self):
        years = 29
        months = 11
        days = 14
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.775740
        actual = futures.conversion_factor('UB', coupon=4.375 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_not_support_error(self):
        with self.assertRaises(NotImplementedError):
            futures.conversion_factor('?', coupon=4.375 / 100, time_to_maturity=1.5)


class TestEligableForDeliveryOf(unittest.TestCase):
    def test_single_time_to_maturity(self):
        time_to_maturity = 8.5
        actual = futures.eligible_for_delivery_in(time_to_maturity)
        expected = 'ZN'

        self.assertEqual(expected, actual)

    def test_array_of_time_to_maturity(self):
        time_to_maturity = np.array([5, 7, 17, 26])
        actual = futures.eligible_for_delivery_in(time_to_maturity).tolist()
        expected = ['ZF', 'ZN', 'ZB', 'UB']

        self.assertEqual(expected, actual)

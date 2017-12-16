import unittest

from fixed_income import futures


class TestConversionFactor(unittest.TestCase):
    def test_2_year(self):
        years = 1
        months = 11
        days = 29
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.906258
        actual = futures.conversion_factor('TU', coupon=0.75 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_3_year(self):
        years = 3
        months = 0
        days = 14
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.867956
        actual = futures.conversion_factor('3YR', coupon=1.125 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_5_year(self):
        years = 4
        months = 11
        days = 29
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.837079
        actual = futures.conversion_factor('FV', coupon=2.125 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_10_year(self):
        years = 9
        months = 5
        days = 14
        time_to_maturity = years + months / 12 + days / 365

        expected = 0.815653
        actual = futures.conversion_factor('TY', coupon=3.375 / 100, time_to_maturity=time_to_maturity)

        self.assertAlmostEqual(expected, actual, places=6)

    def test_classic_bonds(self):
        years = 20
        months = 11
        days = 13
        time_to_maturity = years + months / 12 + days / 365

        expected = 1.044053
        actual = futures.conversion_factor('US', coupon=6.375 / 100, time_to_maturity=time_to_maturity)

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

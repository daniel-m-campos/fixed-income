import math
import unittest

from fixed_income import rates


class TestFunctions(unittest.TestCase):
    def test_compute_days_between_american_dates(self):
        actual = rates.compute_days_between('9/20/2017', '9/13/2018')
        expected = 358
        self.assertEqual(actual, expected)

    def test_bond_equivalent_yield(self):
        actual = rates.bond_equivalent_yield(discount_yield=1.135 / 100, days_to_maturity=168)
        expected = 1.156891557 / 100
        self.assertAlmostEqual(actual, expected)

    def test_treasury_bill_price(self):
        actual = rates.treasury_bill_price(discount_yield=1.135 / 100, days_to_maturity=168)
        expected = 99.47033333
        self.assertAlmostEqual(actual, expected)

    def test_spot_rate_from_discount_factor(self):
        expected = 0.04
        freq = 2
        term = 1
        df = math.pow(1 + expected / freq, -freq * term)
        actual = rates.spot_rate_from_discount_factor(discount_factor=df, term=term, freq=freq)
        self.assertAlmostEqual(actual, expected)

    def test_discount_factor_from_spot_rate(self):
        spot_rate = 0.05
        freq = 3
        term = 2
        expected = math.pow(1 + spot_rate / freq, -freq * term)
        actual = rates.discount_factor_from_spot_rate(spot_rate=spot_rate, term=term, freq=freq)
        self.assertAlmostEqual(actual, expected)

    def test_forward_rate_from_spot(self):
        def df(spot_rate, term, freq):
            return math.pow(1 + spot_rate / freq, -freq * term)

        terms = (0.5, 1.0)
        spot_rates = (0.04, 0.06)
        freq = 4
        discount_factor_1 = df(spot_rates[0], terms[0], freq)
        discount_factor_2 = df(spot_rates[1], terms[1], freq)
        forward_term, forward_rate = rates.forward_rate_from_spot_rates(term_1=terms[0],
                                                                        spot_rate_1=spot_rates[0],
                                                                        term_2=terms[1],
                                                                        spot_rate_2=spot_rates[1],
                                                                        freq=freq)
        forward_discount_factor = df(forward_rate, forward_term, freq)
        self.assertAlmostEqual(discount_factor_2, discount_factor_1 * forward_discount_factor)

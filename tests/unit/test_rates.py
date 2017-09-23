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
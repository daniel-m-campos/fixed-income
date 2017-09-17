import unittest

from fixed_income import bonds


class TestFunctions(unittest.TestCase):
    def test_ytm_equals_coupon_rate(self):
        face_value = 100
        coupon = 6.5
        expected = coupon / face_value
        actual = bonds.yield_to_maturity(bond_price=face_value, face_value=face_value, periods=8, coupon=coupon)
        self.assertAlmostEqual(actual, expected)

    def test_price_equals_cash_flows_when_ytm_is_zero(self):
        face_value = 100
        coupon = 6.5
        periods = 8
        expected = coupon * periods + face_value
        actual = bonds.price(ytm=0, face_value=face_value, periods=periods, coupon=coupon)
        self.assertAlmostEqual(actual, expected)


class TestClasses(unittest.TestCase):
    def test_non_int_periods_causes_assertion_error(self):
        with self.assertRaises(AssertionError):
            bonds.Bond(face_value=100, coupon=5, periods=2.5, ytm=0.03)

    def test_price_equals_par_when_coupon_equals_ytm(self):
        face_value = 100
        ytm = 0.07
        coupon = ytm * face_value
        bond = bonds.Bond(face_value=face_value, coupon=coupon, ytm=ytm, periods=12)
        self.assertAlmostEqual(bond.price, face_value)

    def test_duration_of_zero_equals_periods(self):
        expected = 7
        zcb = bonds.ZeroCoupon(face_value=100, periods=expected, ytm=0.05)
        self.assertAlmostEqual(zcb.duration, expected)

    def test_from_ytm(self):
        face_value = 100
        coupon = 6.5
        periods = 8
        ytm = coupon / face_value
        actual = bonds.Bond.from_price(bond_price=face_value, face_value=face_value, periods=periods, coupon=coupon)
        expected = bonds.Bond(ytm=ytm, face_value=face_value, periods=periods, coupon=coupon)
        self.assertEqual(actual, expected)

    def test_cashflow_iteration(self):
        face_value = 100
        coupon = 7
        periods = 2
        bond = bonds.Bond(ytm=0.01, face_value=face_value, periods=periods, coupon=coupon)
        actual = list(bond)
        expected = [(1, coupon), (2, coupon + face_value)]
        self.assertEqual(actual, expected)

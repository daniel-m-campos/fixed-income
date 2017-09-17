import unittest

from fixed_income import bonds


class TestBonds(unittest.TestCase):
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

import math
import unittest

import pandas as pd

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

    def test_can_bootstrap_is_true(self):
        portfolio = [bonds.TreasuryNote(coupon_rate=0.05, maturity_years=t / 2, annual_ytm=0.05) for t in range(1, 5)]
        self.assertTrue(bonds.can_bootstrap(portfolio))

    def test_can_bootstrap_is_false(self):
        portfolio = [bonds.TreasuryNote(coupon_rate=0.05, maturity_years=0.0, annual_ytm=0.02),
                     bonds.TreasuryNote(coupon_rate=0.05, maturity_years=1.0, annual_ytm=0.02)]
        self.assertFalse(bonds.can_bootstrap(portfolio))

    def test_cash_flows(self):
        portfolio = [bonds.TreasuryNote(coupon_rate=0.05, maturity_years=0.5, annual_ytm=0.02),
                     bonds.TreasuryNote(coupon_rate=0.05, maturity_years=1.0, annual_ytm=0.02),
                     bonds.TreasuryNote(coupon_rate=0.05, maturity_years=2.0, annual_ytm=0.02)]
        actual = bonds.cash_flows(portfolio)
        expected = [[102.5, 0.0, 0.0, 0.0], [2.5, 102.5, 0.0, 0.0], [2.5, 2.5, 2.5, 102.5]]
        self.assertEqual(actual, expected)

    def test_bootstrap(self):
        ytm = 0.1
        periods = list(range(1, 5))
        portfolio = [bonds.TreasuryNote(coupon_rate=ytm, maturity_years=n / 2, annual_ytm=ytm) for n in periods]
        actuals = bonds.bootstrap(portfolio)
        expecteds = (bonds.Zero(face_value=1.0, periods=n, ytm=bonds.period_ytm(ytm)) for n in periods)
        self.assertTrue(all(actual == expected for actual, expected in zip(actuals, expecteds)))


class TestCouponBond(unittest.TestCase):
    def test_non_int_periods_causes_assertion_error(self):
        with self.assertRaises(AssertionError):
            bonds.CouponBond(face_value=100, coupon=5, periods=2.5, ytm=0.03)

    def test_price_equals_par_when_coupon_equals_ytm(self):
        face_value = 100
        ytm = 0.07
        coupon = ytm * face_value
        bond = bonds.CouponBond(face_value=face_value, coupon=coupon, ytm=ytm, periods=12)
        self.assertAlmostEqual(bond.price, face_value)

    def test_duration_of_zero_equals_periods(self):
        expected = 7
        zcb = bonds.Zero(face_value=100, periods=expected, ytm=0.05)
        self.assertAlmostEqual(zcb.macaulay_duration, expected)

    def test_from_price(self):
        face_value = 100
        coupon = 6.5
        periods = 8
        ytm = coupon / face_value
        actual = bonds.CouponBond.from_price(bond_price=face_value, face_value=face_value, periods=periods,
                                             coupon=coupon)
        expected = bonds.CouponBond(ytm=ytm, face_value=face_value, periods=periods, coupon=coupon)
        self.assertEqual(actual, expected)

    def test_from_dataframe(self):
        bond_details = {'bond_price': 100, 'coupon': 5, 'face_value': 100, 'periods': 2}
        df = pd.DataFrame([bond_details], )
        actual = next(bonds.CouponBond.from_dataframe(df))
        expected = bonds.CouponBond.from_price(**bond_details)
        self.assertEqual(actual, expected)

    def test_cash_flow_iteration(self):
        face_value = 100
        coupon = 7
        periods = 2
        bond = bonds.CouponBond(ytm=0.01, face_value=face_value, periods=periods, coupon=coupon)
        actual = list(bond)
        expected = [(1, coupon), (2, coupon + face_value)]
        self.assertEqual(actual, expected)

    def test_macaulay_duration(self):
        """Based question 23a chapter 16 of Bodie Kane Marcus - Investments (10th Ed)"""
        bond = bonds.CouponBond(ytm=0.07, face_value=100, periods=10, coupon=7)
        expected = 7.51523225
        self.assertAlmostEqual(bond.macaulay_duration, expected)

    def test_ytm_convexity(self):
        """Based question 23b chapter 16 of Bodie Kane Marcus - Investments (10th Ed)"""
        bond = bonds.CouponBond(ytm=0.07, face_value=100, periods=10, coupon=7)
        expected = 64.9329593
        self.assertAlmostEqual(bond.ytm_convexity, expected)

    def test_duration(self):
        ytm = 0.05
        coupon = 6
        periods = 2
        face_value = 100
        bond = bonds.CouponBond(face_value, coupon, periods, ytm)
        zero_prices = [math.pow(1 + ytm, -i) for i in range(1, periods + 1)]
        expected = sum(t * c * z for (t, c), z in zip(bond, zero_prices)) / bond.price
        self.assertAlmostEqual(bond.duration, expected)

    def test_price_change_without_ytm_convexity(self):
        """Based question 23c chapter 16 of Bodie Kane Marcus - Investments (10th Ed)"""
        bond = bonds.CouponBond(ytm=0.07, face_value=100, periods=10, coupon=7)
        actual = bond.price_change(ytm_change=0.01)
        expected = -7.02358154
        self.assertAlmostEqual(actual, expected)

    def test_price_change_with_ytm_convexity(self):
        """Based question 23d chapter 16 of Bodie Kane Marcus - Investments (10th Ed)"""
        bond = bonds.CouponBond(ytm=0.07, face_value=100, periods=10, coupon=7)
        actual = bond.price_change(ytm_change=0.01, use_convexity=True)
        expected = -6.69891674
        self.assertAlmostEqual(actual, expected)


class TestZero(unittest.TestCase):
    def test_duration(self):
        maturity = 5
        zero = bonds.Zero(face_value=1, periods=maturity, ytm=0.05)
        self.assertEqual(zero.duration, maturity)


class TestPerpetuity(unittest.TestCase):
    def test_macaulay_duration(self):
        ytm = 0.07
        perpetuity = bonds.Perpetuity(ytm=ytm, coupon=1)
        expected = (1 + ytm) / ytm
        self.assertAlmostEqual(perpetuity.macaulay_duration, expected)

    def test_convexity(self):
        ytm = 0.07
        perpetuity = bonds.Perpetuity(ytm=ytm, coupon=1)
        expected = 2 / ytm ** 2
        self.assertAlmostEqual(perpetuity.ytm_convexity, expected)


class TestTreasuryNote(unittest.TestCase):
    def test_price(self):
        """ Example 2.11 from Pietro Veronesi - Fixed Income Securities"""
        note = bonds.TreasuryNote(coupon_rate=0.0475, maturity_years=9.5, annual_ytm=0.037548)
        expected = 107.8906
        self.assertAlmostEqual(note.price, expected, places=3)

    def test_from_price(self):
        """ Example 2.11 from Pietro Veronesi - Fixed Income Securities"""
        note = bonds.TreasuryNote.from_price(bond_price=141.5267, coupon_rate=0.08875, maturity_years=9.5)
        expected = bonds.period_ytm(0.036603)
        self.assertAlmostEqual(note.ytm, expected, places=6)


class TestFloatingRateBond(unittest.TestCase):
    def test_price_with_zero_spread(self):
        """ Example 2.13 from Pietro Veronesi - Fixed Income Securities"""
        floater = bonds.FloatingRateBond(maturity_years=1, interest_rate=0.06, spread_rate=0)
        expected = floater.face_value
        self.assertAlmostEqual(floater.price, expected)

    def test_price_with_nonzero_spread(self):
        floater = bonds.FloatingRateBond(maturity_years=1, interest_rate=0.0, spread_rate=0.01)
        expected = floater.face_value + floater.periods * floater.fixed_coupon
        self.assertAlmostEqual(floater.price, expected)

    def test_periods_after_reset(self):
        floater = bonds.FloatingRateBond(maturity_years=3.5, interest_rate=0.05, spread_rate=0.01)
        period = 3
        expected = floater.periods - period
        floater.reset(period=period, interest_rate=0.07)
        self.assertAlmostEqual(floater.periods, expected)

    def test_coupon(self):
        floater = bonds.FloatingRateBond(maturity_years=1, interest_rate=0.05, spread_rate=0.01)
        expected = (floater.interest_rate + floater.spread_rate) * floater.face_value / floater.freq
        self.assertAlmostEqual(floater.coupon, expected)

    def test_coupon_after_reset(self):
        floater = bonds.FloatingRateBond(maturity_years=1.5, interest_rate=0.05, spread_rate=0.01)
        new_interest_rate = 0.1
        floater.reset(period=1, interest_rate=new_interest_rate)
        expected = (new_interest_rate + floater.spread_rate) * floater.face_value / floater.freq
        self.assertAlmostEqual(floater.coupon, expected)

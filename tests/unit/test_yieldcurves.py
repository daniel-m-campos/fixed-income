import os
from unittest import TestCase

import numpy as np
import pandas as pd

from fixed_income import yieldcurves

DIR = os.path.dirname(os.path.abspath(__file__))
TIPS_FILE = f'{DIR}/resources/DataTIPS.xlsx'
VASICEK_FILE = f'{DIR}/resources/VasicekData.xlsx'


class TestNelsonSiegelFit(TestCase):
    def setUp(self):
        super().setUp()
        os.path.dirname(os.path.abspath(__file__))
        self.quotes = pd.read_excel(TIPS_FILE, sheetname='Treasury_Quotes')
        self.quotes.index = self.quotes['Time To Maturity'].values
        self.cashflows = pd.read_excel(TIPS_FILE, sheetname='Treasury_Cashflows')
        self.cf_maturities = pd.read_excel(TIPS_FILE, sheetname='Treasury_Cashflows_Maturity')

    def test_fit(self):
        prices = (self.quotes['Bid Price'] + self.quotes['Ask Price']) / 2
        result = yieldcurves.ns_fit(prices, self.cashflows, self.cf_maturities)
        actual = result.x
        expected = np.array([0.03935294, -0.02175923, -0.07813487, 1.91292469])
        self.assertTrue(all(np.isclose(actual, expected)))


class TestVasicekFit(TestCase):
    def setUp(self):
        super().setUp()
        self.quotes = pd.read_excel(VASICEK_FILE, sheetname='Quotes')
        self.quotes.index = range(1, len(self.quotes) + 1)
        self.cashflows = pd.read_excel(VASICEK_FILE, sheetname='CashFlows')
        self.cf_maturities = pd.read_excel(VASICEK_FILE, sheetname='Maturities')

    def test_fit(self):
        r0 = 0.011499737607216544
        sigma = 0.032261681963642166
        prices = (self.quotes['Bid'] + self.quotes['Ask']) / 2 + self.quotes['AccruedInterest']
        result = yieldcurves.vasicek_fit(r0, sigma, prices, self.cashflows, self.cf_maturities)
        actual = result.x
        expected = np.array([0.01768098, 0.2130499])
        self.assertTrue(all(np.isclose(actual, expected)))

import os
from unittest import TestCase

import numpy as np
import pandas as pd

from fixed_income import yieldcurves

DATA_FILE = f'{os.path.dirname(os.path.abspath(__file__))}/resources/DataTIPS.xlsx'


class TestFit(TestCase):
    def setUp(self):
        super().setUp()
        os.path.dirname(os.path.abspath(__file__))
        self.quotes = pd.read_excel(DATA_FILE, sheetname='Treasury_Quotes')
        self.quotes.index = self.quotes['Time To Maturity'].values
        self.cashflows = pd.read_excel(DATA_FILE, sheetname='Treasury_Cashflows')
        self.cf_maturities = pd.read_excel(DATA_FILE, sheetname='Treasury_Cashflows_Maturity')

    def test_fit(self):
        prices = (self.quotes['Bid Price'] + self.quotes['Ask Price']) / 2
        result = yieldcurves.fit(prices, self.cashflows, self.cf_maturities)
        actual = result.x
        expected = np.array([0.03935264, -0.02175932, -0.07813865, 1.91278737])
        self.assertTrue(all(np.isclose(actual, expected)))

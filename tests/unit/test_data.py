import datetime
import unittest

import vcr

from fixed_income import data


class TestTreasuryDirect(unittest.TestCase):
    @vcr.use_cassette('resources/test_load_data_from_date.yml')
    def test_load_data_from_date(self):
        df = data.treasury_direct(datetime.date(year=2017, month=11, day=17))
        self.assertTrue(~df.empty)

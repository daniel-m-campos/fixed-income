import math

from pandas import Timestamp


def compute_days_between(start_date, end_date):
    """Computes number of days between dates"""
    time_delta = Timestamp(end_date) - Timestamp(start_date)
    return time_delta.days


def treasury_bill_price(discount_yield, days_to_maturity):
    """Computes price ot treasury bill"""
    return 100 * (1 - days_to_maturity / 360 * discount_yield)


def bond_equivalent_yield(discount_yield, days_to_maturity):
    """Computes bond equivalent yield from treasury bill discount yield"""
    return 365 * discount_yield / (360 - discount_yield * days_to_maturity)


def discount_factor_from_short_rate(short_rate, term):
    return 1 / (1 + short_rate * term)


def forward_rate_from_short_rates(term_1, short_rate_1, term_2, short_rate_2):
    """Computes term, and forward rate for pair of short rates"""
    term = term_2 - term_1
    df_1 = discount_factor_from_short_rate(short_rate_1, term_1)
    df_2 = discount_factor_from_short_rate(short_rate_2, term_2)
    return term, df_2 / df_1 - 1


def discount_factor_from_spot_rate(spot_rate, term, freq):
    """Computes discount factor from spot rate"""
    return math.pow(1 + spot_rate / freq, -freq * term)


def spot_rate_from_discount_factor(discount_factor, term, freq):
    """Computes spot rate from discount factor"""
    return freq * (math.pow(discount_factor, -1 / (freq * term)) - 1)


def forward_rate_from_spot_rates(term_1, spot_rate_1, term_2, spot_rate_2, freq=2):
    """Computes term, and forward rate for pair of spot rates"""
    term = term_2 - term_1
    df_1 = discount_factor_from_spot_rate(spot_rate_1, term_1, freq)
    df_2 = discount_factor_from_spot_rate(spot_rate_2, term_2, freq)
    forward_df = df_2 / df_1
    return term, spot_rate_from_discount_factor(discount_factor=forward_df, term=term, freq=freq)

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

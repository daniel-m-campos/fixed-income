import math

import numpy as np
import pandas as pd
from pandas import Timestamp

__all__ = [
    'compute_days_between',
    'treasury_bill_price',
    'bond_equivalent_yield',
    'discount_factor_from',
    'spot_rate_from',
    'forward_rate_from'
]


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


def is_valid_freq(freq):
    return math.isfinite(freq) and isinstance(freq, int) and freq > 0


def discount_factor_from(spot_rate, term, freq=math.inf):
    """Computes discount factor from spot rate

    The  discount factor from a spot rate given term and compounding frequency.

    Parameters
    ----------
    spot_rate : float
        The spot rate
    term : float
        The term or time to maturity
    freq : int or math.inf, optional
        The compounding frequency, default is math.inf which results in continuous compounding rates.

    Returns
    -------
    float
        The discount factor determined the spot rate.

    Raises
    ------
    ValueError
        If freq is not math.inf of positive int

    """
    if freq == math.inf:
        return math.exp(-spot_rate * term)
    elif is_valid_freq(freq):
        return math.pow(1 + spot_rate / freq, -freq * term)
    else:
        raise ValueError('Freq must be math.inf or positive int')


def spot_rate_from(discount_factor, term, freq=math.inf):
    """Computes spot rate from discount factor

    The spot rate from the discount factor given term and compounding frequency.

    Parameters
    ----------
    discount_factor : float or pandas.Series
        The discount factor to determine the spot rate.
    term : float
        The term or time to maturity
    freq : int or math.inf, optional
        The compounding frequency, default is math.inf which results in continuous compounding rates.

    Returns
    -------
    float or pandas.Series
        The spot rate

    Raises
    ------
    ValueError
        If freq is not math.inf of positive int

    """
    if freq == math.inf:
        package = math
        if isinstance(discount_factor, pd.Series):
            package = np
        return -1 / term * package.log(discount_factor)
    elif is_valid_freq(freq):
        return freq * (math.pow(discount_factor, -1 / (freq * term)) - 1)
    else:
        raise ValueError('Freq must be math.inf or positive int')


def forward_rate_from(rate_1, term_1, rate_2, term_2, freq=math.inf):
    """Computes forward rate from pair of spot rates

    The forward rate between pair of spot rates given terms and compounding frequency.

    Parameters
    ----------
    rate_1 : float
       The first spot rate
    term_1 : float
        The first term or time to maturity
    rate_2 : float
       The second spot rate
    term_2 : float
        The second term or time to maturity
    freq : int or math.inf, optional
        The compounding frequency, default is math.inf which results in continuous compounding rates.

    Returns
    -------
    float, float
        The term and rate of the forward

    Raises
    ------
    ValueError
        If freq is not math.inf of positive int

    """
    term = term_2 - term_1
    if freq == math.inf:
        return term, (rate_2 * term_2 - rate_1 * term_1) / term
    elif is_valid_freq(freq):
        df_1 = discount_factor_from(rate_1, term_1, freq)
        df_2 = discount_factor_from(rate_2, term_2, freq)
        forward_df = df_2 / df_1
        return term, spot_rate_from(discount_factor=forward_df, term=term, freq=freq)
    else:
        raise ValueError('Freq must be math.inf or positive int')

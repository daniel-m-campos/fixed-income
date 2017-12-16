import numpy as np

MONTHS_IN_QUARTER = 3
MONTHS_IN_YEAR = 12
CONVERSION_YIELD = 0.06
CONVERSION_FV = 1.03


def _n_and_v(globex_code, year_fraction):
    mask = np.in1d(globex_code, ('ZN', 'ZB', 'UB'))
    n = mask * np.nan
    v = mask * np.nan
    if mask.any():
        n[mask] = np.floor(year_fraction / MONTHS_IN_QUARTER * MONTHS_IN_YEAR) * MONTHS_IN_QUARTER
        v[mask] = n * (n < 7) + MONTHS_IN_QUARTER * (n >= 7)

    mask = np.in1d(globex_code, ('ZT', 'Z3N', 'ZF'))
    if mask.any():
        n[mask] = np.floor(year_fraction * MONTHS_IN_YEAR)
        v[mask] = n * (n < 7) + (n - 6) * (n >= 7)

    if np.isnan(n).all():
        raise NotImplementedError('No supported value of globex_code found!')

    return n, v


def conversion_factor(globex_code, coupon, time_to_maturity):
    years = np.floor(time_to_maturity)
    year_fraction = time_to_maturity - years
    n, v = _n_and_v(globex_code, year_fraction)
    a = 1 / pow(CONVERSION_FV, v / 6)
    b = (coupon / 2) * (6 - v) / 6
    c = 1 / pow(CONVERSION_FV, 2 * years if n < 7 else 2 * years + 1)
    d = (coupon / CONVERSION_YIELD) * (1 - c)

    factor = a * (coupon / 2 + c + d) - b
    return factor if factor.size > 1 else float(factor)


def eligible_for_delivery_in(time_to_maturity):
    tau = np.array(time_to_maturity)
    globax_codes = np.empty(tau.shape, dtype='<U3')

    mask = tau > 25.0
    globax_codes[mask] = 'UB'
    mask = (15.0 <= tau) & (tau < 25.0)
    globax_codes[mask] = 'ZB'
    mask = (9.5 <= tau) & (tau < 10.0)
    globax_codes[mask] = 'TN'
    mask = (6.5 <= tau) & (tau < 10.0)
    globax_codes[mask] = 'ZN'
    mask = (6.5 <= tau) & (tau < 10.0)
    globax_codes[mask] = 'ZN'
    mask = (4 + 2 / 12 <= tau) & (tau < 5 + 3 / 12)
    globax_codes[mask] = 'ZF'
    mask = (2 + 9 / 12 <= tau) & (tau < 4 + 2 / 12)
    globax_codes[mask] = 'Z3N'
    mask = (1 + 9 / 12 <= tau) & (tau < 2 + 9 / 12)
    globax_codes[mask] = 'ZT'

    return globax_codes if globax_codes.size > 1 else str(globax_codes)

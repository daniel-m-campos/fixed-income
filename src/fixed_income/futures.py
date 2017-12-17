import numpy as np
import pandas as pd

MONTHS_IN_QUARTER = 3
MONTHS_IN_YEAR = 12
CONVERSION_YIELD = 0.06
CONVERSION_FV = 1.03

GLOBEX_CODES = ('ZN', 'ZB', 'UB', 'ZT', 'Z3N', 'ZF')


def _n_and_v(globex_code, year_fraction):
    mask = np.in1d(globex_code, ('ZN', 'ZB', 'UB'))
    n = mask * np.nan
    v = mask * np.nan
    if mask.any():
        values = np.floor(year_fraction / MONTHS_IN_QUARTER * MONTHS_IN_YEAR) * MONTHS_IN_QUARTER
        n[mask] = values if isinstance(values, float) else values[mask]
        values = n * (n < 7) + MONTHS_IN_QUARTER * (n >= 7)
        v[mask] = values if isinstance(values, float) else values[mask]

    mask = np.in1d(globex_code, ('ZT', 'Z3N', 'ZF'))
    if mask.any():
        values = np.floor(year_fraction * MONTHS_IN_YEAR)
        n[mask] = values if isinstance(values, float) else values[mask]
        values = n * (n < 7) + (n - 6) * (n >= 7)
        v[mask] = values if isinstance(values, float) else values[mask]

    if np.isnan(n).all():
        raise NotImplementedError('No supported value of globex_code found!')

    return n, v


def conversion_factor(globex_code, coupon, time_to_maturity):
    years = np.floor(time_to_maturity)
    year_fraction = time_to_maturity - years
    n, v = _n_and_v(globex_code, year_fraction)
    a = 1 / pow(CONVERSION_FV, v / 6)
    b = (coupon / 2) * (6 - v) / 6
    c = 1 / pow(CONVERSION_FV, 2 * years + 1 * (n >= 7))
    d = (coupon / CONVERSION_YIELD) * (1 - c)

    factor = a * (coupon / 2 + c + d) - b
    return factor if factor.size > 1 else float(factor)


def extract_deliverables(df):
    deliverables = []
    for code in GLOBEX_CODES:
        tmp_df = df[find_deliverables_of(code, df.MATURITY)].copy()
        tmp_df['DELIVERABLE'] = code
        tmp_df['CONV_FACTOR'] = conversion_factor(tmp_df.DELIVERABLE, tmp_df.COUPON / 100, tmp_df.MATURITY)
        deliverables.append(tmp_df)

    deliverables = pd.concat(deliverables)
    deliverables.index = np.arange(len(deliverables))
    return deliverables


def find_deliverables_of(globex_code, time_to_maturity):
    tau = np.array(time_to_maturity)
    if globex_code == 'UB':
        return tau >= 25.0
    elif globex_code == 'ZB':
        return (15.0 <= tau) & (tau < 25.0)
    elif globex_code == 'ZN':
        return (6 + 6 / 12 <= tau) & (tau <= 10.0)
    elif globex_code == 'TN':
        return (9 + 5 / 12 <= tau) & (tau <= 10.0)
    elif globex_code == 'ZF':
        return (4 + 2 / 12 <= tau) & (tau <= 5 + 3 / 12)
    elif globex_code == 'Z3N':
        return (2 + 9 / 12 <= tau) & (tau <= 5 + 3 / 12)
    elif globex_code == 'ZT':
        return (1 + 9 / 12 <= tau) & (tau <= 5 + 3 / 12)
    else:
        raise NotImplementedError(f'{globex_code} not supported!')

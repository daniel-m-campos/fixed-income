import numpy as np

MONTHS_IN_QUARTER = 3
MONTHS_IN_YEAR = 12
CONVERSION_YIELD = 0.06
CONVERSION_FV = 1.03


def conversion_factor(globex_code, coupon, time_to_maturity):
    years = np.floor(time_to_maturity)
    year_fraction = time_to_maturity - years
    if globex_code in ('ZN', 'ZB', 'UB'):
        months = np.floor(year_fraction / MONTHS_IN_QUARTER * MONTHS_IN_YEAR) * MONTHS_IN_QUARTER
        v = months if months < 7 else MONTHS_IN_QUARTER
    elif globex_code in ('ZT', 'Z3N', 'ZF'):
        months = np.floor(year_fraction * MONTHS_IN_YEAR)
        v = months if months < 7 else months - 6
    else:
        raise NotImplementedError(f'{globex_code} not supported.')
    a = 1 / pow(CONVERSION_FV, v / 6)
    b = (coupon / 2) * (6 - v) / 6
    c = 1 / pow(CONVERSION_FV, 2 * years if months < 7 else 2 * years + 1)
    d = (coupon / CONVERSION_YIELD) * (1 - c)

    return a * (coupon / 2 + c + d) - b


def eligible_for_delivery_of(time_to_maturity):
    if time_to_maturity > 25.0:
        return 'UB'
    elif 15.0 <= time_to_maturity < 25.0:
        return 'ZB'
    elif 9.5 <= time_to_maturity < 10.0:
        return 'TN'
    elif 6.5 <= time_to_maturity < 10.0:
        return 'ZN'
    elif 4 + 2 / 12 <= time_to_maturity < 5 + 3 / 12:
        return 'ZF'
    elif 2 + 9 / 12 <= time_to_maturity < 5 + 3 / 12:
        return 'Z3N'
    elif 1 + 9 / 12 <= time_to_maturity < 5 + 3 / 12:
        return 'ZT'

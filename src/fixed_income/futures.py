import numpy as np

CONVERSION_YIELD = 0.06
CONVERSION_FV = 1.03


def conversion_factor(security_type, coupon, time_to_maturity):
    years = np.floor(time_to_maturity)
    year_fraction = time_to_maturity - years
    if security_type in ('TY', 'US', 'UB'):
        months = np.floor(year_fraction / 3 * 12) * 3
        v = months if months < 7 else 3
    else:
        months = np.floor(year_fraction * 12)
        v = months if months < 7 else months - 6
    a = 1 / pow(CONVERSION_FV, v / 6)
    b = (coupon / 2) * (6 - v) / 6
    c = 1 / pow(CONVERSION_FV, 2 * years if months < 7 else 2 * years + 1)
    d = (coupon / CONVERSION_YIELD) * (1 - c)

    return a * (coupon / 2 + c + d) - b

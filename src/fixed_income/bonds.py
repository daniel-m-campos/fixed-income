import math

import numpy as np
import pandas as pd
import scipy.optimize as optimize


def future_value_factor(ytm, periods):
    return math.pow(1 + ytm, periods)


def present_value_factor(ytm, periods):
    return future_value_factor(ytm, -periods)


def price(face_value, coupon, periods, ytm):
    """The clean price of a coupon paying bond"""
    pv_factor = present_value_factor(ytm, periods)
    annuity_factor = 1 / ytm * (1 - pv_factor) if ytm != 0.0 else periods
    return coupon * annuity_factor + face_value * pv_factor


def yield_to_maturity(bond_price, face_value, periods, coupon, guess=0.05):
    return optimize.newton(lambda ytm: price(face_value, coupon, periods, ytm) - bond_price, guess)


def to_periods(maturity_years, freq=2):
    return int(maturity_years * freq)


def to_coupon(par, coupon_rate, freq=2):
    return par * coupon_rate / freq


def period_ytm(annual_ytm, freq=2):
    return annual_ytm / freq


def can_bootstrap(portfolio):
    periods = set(bond.periods for bond in portfolio)
    return periods == set(range(1, max(periods) + 1))


def cash_flows(portfolio):
    longest_bond = max(portfolio, key=lambda bond: bond.periods)
    cfs = [[cf for _, cf in bond] for bond in portfolio]
    return [cf + [0.0] * (longest_bond.periods - len(cf)) for cf in cfs]


def bootstrap(portfolio):
    assert can_bootstrap(portfolio), 'Bonds in portfolio cannot be bootstrapped'
    prices = np.row_stack([bond.price for bond in portfolio])
    cfs = np.matrix(cash_flows(portfolio))
    dfs = cfs.getI() @ prices
    dfs = np.array(dfs).reshape(-1).tolist()
    return [Zero.from_price(bond_price=df, periods=n, face_value=1.0) for n, df in enumerate(dfs, start=1)]


def to_dataframe(portfolio):
    df = [vars(b) for b in portfolio]
    df = pd.DataFrame(df)
    df.columns = [k[1:] if k.startswith('_') else k for k in df.columns]
    return df


class CouponBond:
    def __init__(self, face_value, coupon, periods, ytm):
        assert isinstance(periods, (int, np.integer)) or periods == math.inf
        self._face_value = face_value
        self._coupon = coupon
        self._periods = periods
        self._ytm = ytm
        self._price = price(self.face_value, self.coupon, self.periods, self.ytm)

    @property
    def face_value(self):
        return self._face_value

    @property
    def coupon(self):
        return self._coupon

    @property
    def periods(self):
        return self._periods

    @property
    def ytm(self):
        return self._ytm

    @property
    def price(self):
        return self._price

    @property
    def macaulay_duration(self):
        weighted_cash_flow = sum(t * cash_flow * present_value_factor(self.ytm, t) for t, cash_flow in self)
        return weighted_cash_flow / self.price

    @property
    def duration(self):
        """The percentage change in price given a change in the level on continuous spot rates"""
        # In this case it happens to coincide with the macaulay duration.
        return self.macaulay_duration

    @property
    def modified_duration(self):
        """The percentage change in price given a change in yield to maturity"""
        return self.macaulay_duration / (1 + self.ytm)

    @property
    def ytm_convexity(self):
        weighted_cash_flow = sum(t * (t + 1) * cash_flow * present_value_factor(self.ytm, t) for t, cash_flow in self)
        return weighted_cash_flow / self.price * present_value_factor(self.ytm, 2)

    def __eq__(self, other):
        if isinstance(other, CouponBond):
            return all(math.isclose(self.__dict__[key], other.__dict__[key]) for key in self.__dict__)
        return NotImplemented

    def __iter__(self):
        for t in range(1, self.periods):
            yield t, self.coupon
        yield self.periods, self.coupon + self.face_value

    def __repr__(self):
        property_string = ', '.join('{}={:.7g}'.format(k[1:], v) for k, v in self.__dict__.items())
        return "{}({})".format(self.__class__.__name__, property_string)

    def price_change(self, ytm_change, use_convexity=False):
        sensitivity = -self.modified_duration
        if use_convexity:
            sensitivity += 0.5 * ytm_change * self.ytm_convexity
        return self.price * ytm_change * sensitivity

    @classmethod
    def from_price(cls, bond_price, coupon, periods, face_value):
        ytm = yield_to_maturity(bond_price=bond_price, face_value=face_value, coupon=coupon, periods=periods)
        return cls(face_value=face_value, coupon=coupon, periods=periods, ytm=ytm)

    @classmethod
    def from_dataframe(cls, df):
        assert {'coupon', 'bond_price', 'periods', 'face_value'}.issubset(df.columns)
        for index, row in df.iterrows():
            yield cls.from_price(face_value=row.face_value,
                                 coupon=row.coupon,
                                 periods=row.periods,
                                 bond_price=row.bond_price)


class Zero(CouponBond):
    def __init__(self, face_value, periods, ytm):
        super().__init__(face_value=face_value, coupon=0, periods=periods, ytm=ytm)

    def __iter__(self):
        yield self.periods, self.face_value

    @classmethod
    def from_price(cls, bond_price, face_value, periods, **kwargs):
        ytm = yield_to_maturity(bond_price=bond_price, face_value=face_value, coupon=0.0, periods=periods)
        return cls(face_value, periods, ytm)


class Perpetuity(CouponBond):
    def __init__(self, coupon, ytm):
        super().__init__(face_value=0, coupon=coupon, periods=math.inf, ytm=ytm)

    @property
    def price(self):
        return self.coupon / self.ytm

    @property
    def macaulay_duration(self):
        return (1 + self.ytm) / self.ytm

    @property
    def ytm_convexity(self):
        return 2 / math.pow(self.ytm, 2)


class TreasuryNote(CouponBond):
    _par = 100.0
    _freq = 2

    def __init__(self, coupon_rate, maturity_years, annual_ytm):
        super().__init__(face_value=self._par,
                         coupon=to_coupon(self._par, coupon_rate, self._freq),
                         periods=to_periods(maturity_years, self._freq),
                         ytm=period_ytm(annual_ytm))

    @property
    def duration(self):
        period_duration = super().duration
        return period_duration / self._freq

    @property
    def freq(self):
        return self._freq

    @property
    def annual_ytm(self):
        return self._freq * self._ytm

    @classmethod
    def from_price(cls, bond_price, coupon_rate, maturity_years, **kwargs):
        semi_annual_ytm = yield_to_maturity(bond_price=bond_price,
                                            face_value=cls._par,
                                            coupon=to_coupon(cls._par, coupon_rate, cls._freq),
                                            periods=to_periods(maturity_years, cls._freq))
        return cls(coupon_rate, maturity_years, semi_annual_ytm * cls._freq)

    @classmethod
    def from_dataframe(cls, df):
        assert {'coupon_rate', 'bond_price', 'maturity_years'}.issubset(df.columns)
        for index, row in df.iterrows():
            yield cls.from_price(bond_price=row.bond_price,
                                 coupon_rate=row.coupon_rate,
                                 maturity_years=row.maturity_years)


class FloatingRateBond:
    def __init__(self, maturity_years, interest_rate, spread_rate=0, freq=1, face_value=100):
        self._maturity_years = maturity_years
        self._freq = freq
        self._face_value = face_value
        self._periods = to_periods(maturity_years, self._freq)
        self._interest_rate = interest_rate
        self._spread_rate = spread_rate
        self._fixed_bond = CouponBond(face_value=0,
                                      coupon=to_coupon(self._face_value, spread_rate, self._freq),
                                      periods=self._periods,
                                      ytm=period_ytm(interest_rate, self._freq))

    def reset(self, period, interest_rate):
        self._periods -= period
        self._interest_rate = interest_rate
        self._fixed_bond = CouponBond(face_value=0,
                                      coupon=to_coupon(self._par, self._spread_rate, self._freq),
                                      periods=self._periods,
                                      ytm=period_ytm(interest_rate, self._freq))

    @property
    def face_value(self):
        return self._par

    @property
    def freq(self):
        return self._freq

    @property
    def periods(self):
        return self._periods

    @property
    def maturity_years(self):
        return self._maturity_years

    @property
    def interest_rate(self):
        return self._interest_rate

    @property
    def spread_rate(self):
        return self._spread_rate

    @property
    def fixed_coupon(self):
        return self._fixed_bond.coupon

    @property
    def coupon(self):
        return self._fixed_bond.coupon + to_coupon(self._par, self._interest_rate, self._freq)

    @property
    def price(self):
        return self._face_value + self._fixed_bond.price

    @property
    def duration(self):
        return 1 / self._freq


class InverseFloatingRateBond:
    _par = 100

    def __init__(self, price, duration, convexity, leverage):
        self.price = price
        self.duration = duration
        self.convexity = convexity
        self.leverage = leverage

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return vars(self) == vars(other)
        else:
            return NotImplemented

    def __repr__(self):
        property_string = ', '.join('{}={:.7g}'.format(k, v) for k, v in vars(self).items())
        return "{}({})".format(self.__class__.__name__, property_string)

    @classmethod
    def from_components(cls, coupon_bond, zero_bond, leverage=1):
        price = zero_bond.price * leverage + coupon_bond.price - cls._par * leverage
        duration = (zero_bond.price * leverage * zero_bond.duration
                    + coupon_bond.price * coupon_bond.duration
                    - cls._par * leverage) / price
        convexity = (zero_bond.price * leverage * zero_bond.convexity
                     + coupon_bond.price * coupon_bond.convexity
                     - cls._par * leverage) / price
        return cls(price, duration, convexity)

    @classmethod
    def from_zeros(cls, zeros, fixed_coupon, maturity, leverage, freq=2):
        assert len(zeros) == freq * maturity
        payments = fixed_coupon / freq * np.ones((maturity * freq,))
        payments[-1] += cls._par
        price_fixed = sum(p * z for p, z in zip(payments, zeros))
        price_float = cls._par
        price_zero = cls._par * zeros[-1]
        price = price_fixed + leverage * (price_zero - price_float)

        duration_fixed = sum(p * t / freq * z for t, (p, z) in enumerate(zip(payments, zeros), start=1))
        duration_fixed /= price_fixed
        duration_float = 1 / freq
        duration_zero = maturity

        duration = (price_zero * leverage * duration_zero
                    + price_fixed * duration_fixed
                    - price_float * leverage * duration_float) / price

        convexity_fixed = sum(p * pow(t / freq, 2) * z for t, (p, z) in enumerate(zip(payments, zeros), start=1))
        convexity_fixed /= price_fixed
        convexity_float = duration_float ** 2
        convexity_zero = duration_zero ** 2

        convexity = (price_zero * leverage * convexity_zero
                     + price_fixed * convexity_fixed
                     - cls._par * leverage * convexity_float) / price

        return cls(price, duration, convexity, leverage)

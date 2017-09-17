import math

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


class CouponBond:
    def __init__(self, face_value, coupon, periods, ytm):
        assert isinstance(periods, int) or periods == math.inf
        self._face_value = face_value
        self._coupon = coupon
        self._periods = periods
        self._ytm = ytm

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
        return price(self.face_value, self.coupon, self.periods, self.ytm)

    @property
    def duration(self):
        weighted_cash_flow = sum(t * cash_flow * present_value_factor(self.ytm, t) for t, cash_flow in self)
        return weighted_cash_flow / self.price

    @property
    def modified_duration(self):
        return self.duration / (1 + self.ytm)

    @property
    def convexity(self):
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

    def price_change(self, ytm_change, use_convexity=False):
        sensitivity = -self.modified_duration
        if use_convexity:
            sensitivity += 0.5 * ytm_change * self.convexity
        return self.price * ytm_change * sensitivity

    @classmethod
    def from_price(cls, bond_price, face_value, coupon, periods):
        ytm = yield_to_maturity(bond_price=bond_price, face_value=face_value, coupon=coupon, periods=periods)
        return cls(face_value=face_value, coupon=coupon, periods=periods, ytm=ytm)


class Zero(CouponBond):
    def __init__(self, face_value, periods, ytm):
        super().__init__(face_value=face_value, coupon=0, periods=periods, ytm=ytm)

    def __iter__(self):
        yield self.periods, self.face_value


class Perpetuity(CouponBond):
    def __init__(self, coupon, ytm):
        super().__init__(face_value=0, coupon=coupon, periods=math.inf, ytm=ytm)

    @property
    def price(self):
        return self.coupon / self.ytm

    @property
    def duration(self):
        return (1 + self.ytm) / self.ytm

    @property
    def convexity(self):
        return 2 / math.pow(self.ytm, 2)


def periods(maturity_years, freq=2):
    return int(maturity_years * freq)


def coupon(par, coupon_rate, freq=2):
    return par * coupon_rate / freq


def period_ytm(annual_ytm, freq=2):
    return annual_ytm / freq


class TreasuryNote(CouponBond):
    _par = 100.0
    _freq = 2

    def __init__(self, coupon_rate, maturity_years, annual_ytm):
        super().__init__(face_value=self._par,
                         coupon=coupon(self._par, coupon_rate, self._freq),
                         periods=periods(maturity_years, self._freq),
                         ytm=period_ytm(annual_ytm))

    @property
    def freq(self):
        return self._freq

    @classmethod
    def from_price(cls, bond_price, coupon_rate, maturity_years, **kwargs):
        semi_annual_ytm = yield_to_maturity(bond_price=bond_price,
                                            face_value=cls._par,
                                            coupon=coupon(cls._par, coupon_rate, cls._freq),
                                            periods=periods(maturity_years, cls._freq))
        return cls(coupon_rate, maturity_years, semi_annual_ytm * cls._freq)


class SemiAnnualFloatingRateBond:
    _par = 100.0
    _freq = 2

    def __init__(self, maturity_years, interest_rate, spread_rate=0):
        self._interest_rate = interest_rate
        self._spread_rate = spread_rate
        self._periods = periods(maturity_years, self._freq)
        self._fixed_bond = CouponBond(face_value=0,
                                      coupon=coupon(self._par, spread_rate, self._freq),
                                      periods=self._periods,
                                      ytm=period_ytm(interest_rate, self._freq))

    def reset(self, period, interest_rate):
        self._interest_rate = interest_rate
        self._fixed_bond = CouponBond(face_value=0,
                                      coupon=coupon(self._par, self._spread_rate, self._freq),
                                      periods=self._periods - period,
                                      ytm=period_ytm(interest_rate, self._freq))

    @property
    def face_value(self):
        return self._par

    @property
    def freq(self):
        return self._freq

    @property
    def periods(self):
        return self._fixed_bond.periods

    @property
    def interest_rate(self):
        return self._interest_rate

    @property
    def spread_coupon(self):
        return self._fixed_bond.coupon

    @property
    def price(self):
        return self._par + self._fixed_bond.price

    @property
    def duration(self):
        # TODO: fill in
        return 0.0

    @property
    def modified_duration(self):
        # TODO: fill in
        return 0.0

    @property
    def convexity(self):
        # TODO: fill in
        return 0.0

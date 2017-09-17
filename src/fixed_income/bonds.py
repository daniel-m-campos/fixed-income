import math

import scipy.optimize as optimize


def future_value_factor(ytm, periods):
    return math.pow(1 + ytm, periods)


def present_value_factor(ytm, periods):
    return future_value_factor(ytm, -periods)


def price(face_value, coupon, periods, ytm):
    pv_factor = present_value_factor(ytm, periods)
    annuity_factor = 1 / ytm * (1 - pv_factor) if ytm != 0.0 else periods
    return coupon * annuity_factor + face_value * pv_factor


def yield_to_maturity(bond_price, face_value, periods, coupon, guess=0.05):
    return optimize.newton(lambda ytm: price(face_value, coupon, periods, ytm) - bond_price, guess)


class Bond:
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
        if isinstance(other, Bond):
            return all(math.isclose(self.__dict__[key], other.__dict__[key]) for key in self.__dict__)
        return NotImplemented

    def __iter__(self):
        for t in range(1, self.periods):
            yield t, self.coupon
        yield self.periods, self.coupon + self.face_value

    @classmethod
    def from_price(cls, bond_price, face_value, coupon, periods):
        ytm = yield_to_maturity(bond_price=bond_price, face_value=face_value, coupon=coupon, periods=periods)
        return cls(face_value=face_value, coupon=coupon, periods=periods, ytm=ytm)


class ZeroCoupon(Bond):
    def __init__(self, face_value, periods, ytm):
        super().__init__(face_value=face_value, coupon=0, periods=periods, ytm=ytm)


class Perpetuity(Bond):
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

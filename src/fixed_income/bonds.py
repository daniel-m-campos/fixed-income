import math

import scipy.optimize as optimize


def future_value_factor(ytm, periods):
    return math.pow(1 + ytm, periods)


def present_value_factor(ytm, periods):
    return future_value_factor(ytm, -periods)


def price(face_value, coupon, periods, ytm):
    assert isinstance(periods, int)
    pv_factor = present_value_factor(ytm, periods)
    annuity_factor = 1 / ytm * (1 - pv_factor) if ytm != 0.0 else periods
    return coupon * annuity_factor + face_value * pv_factor


def yield_to_maturity(bond_price, face_value, periods, coupon, guess=0.05):
    return optimize.newton(lambda ytm: price(face_value, coupon, periods, ytm) - bond_price, guess)


class Bond:
    def __init__(self, face_value, coupon, periods, ytm):
        assert isinstance(periods, int)
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
        weighted_cf = sum(self.coupon * present_value_factor(self.ytm, n) for n in range(1, self.periods + 1))
        weighted_cf += self.periods * self.face_value * present_value_factor(self.ytm, self.periods)
        return weighted_cf / self.price

    @property
    def modified_duration(self):
        return self.duration / (1 + self.ytm)

    @property
    def convexity(self):
        # TODO: complete
        return 0.0

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
        super().__init__(face_value, 0, periods, ytm)


class Perpetuity(Bond):
    def __init__(self, coupon, ytm):
        super().__init__(0, coupon, math.inf, ytm)

    @property
    def price(self):
        return self.coupon / self.ytm

    @property
    def duration(self):
        return (1 + self.ytm) / self.ytm

    @property
    def convexity(self):
        # TODO: complete
        return 0.0

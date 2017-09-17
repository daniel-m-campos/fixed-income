import math


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
        zero_to_maturity = math.pow(1 + self.ytm, -self.periods)
        return self._coupon / self.ytm * (1 - zero_to_maturity) + self.face_value * zero_to_maturity

    @property
    def duration(self):
        weighted_cf = sum(self.coupon / math.pow(1 + self.ytm, n) for n in range(1, self.periods + 1))
        weighted_cf += self.periods * self.face_value / math.pow(1 + self.ytm, self.periods)
        return weighted_cf / self.price

    @property
    def modified_duration(self):
        return self.duration / (1 + self.ytm)

    @property
    def convexity(self):
        # TODO: complete
        return 0.0


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

"""Small timing helper shared by the vision modules."""


class Held:
    """True only after a condition has been true CONTINUOUSLY for min_s seconds (filters short blips).

        tilt = Held(2.0)
        tilted = tilt.update(t, abs(roll) > 20)   # True once the head stays tilted for 2 s
    """

    def __init__(self, min_s):
        self.min_s = min_s
        self.since = None            # when the condition became true (None = false now)

    def update(self, t, active):
        if not active:
            self.since = None
        elif self.since is None:
            self.since = t
        return self.since is not None and t - self.since >= self.min_s

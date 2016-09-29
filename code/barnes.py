# Barnes (2007) gyrochronology relations.

import numpy as np


def age(period, bv):
    """
    From Barnes 2007.
    :param period:
        Rotation period in days.
    :param bv:
        B-V color.
    Returns:
        Age in Gyr.
    """
    a, b, c, n = .7725, .601, .4, .5189
    return (period / a * (bv - c) ** b) ** n


def period(age, bv):
    """
    From Barnes 2007. This is just a place holder - need to update this model.
    age in Gyr.
    Returns period in days.
    """
    a, b, c, n = .7725, .601, .4, .5189
    return a * (bv - c) ** b * (age * 1e3) ** n

if __name__ == "__main__":
    print(period(4.25, .65))
    print(age(26.5, .65))

import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSquaredKernel

def predict_age(t, p):

    # load training data
    kid, mass, merr, age, aerr, z, zerr, period, perr, teff, terr = \
            np.genfromtxt("sun_sample.txt").T

    # set your x, y z and hyperparameters
    D = np.vstack((teff, period)).T
    A, lt, lp = 10, 2000, 50

    # GP prediction
    k = A**2 * ExpSquaredKernel([lt**2, lp**2], ndim=2)
    gp = george.GP(k)
    gp.compute(D, aerr)

    xs = np.zeros((1, 2))
    xs[0, 0] = t
    xs[0, 1] = p

    mu, cov = gp.predict(age, xs)  # do the prediction
    v = np.diag(cov)**.5

    return mu

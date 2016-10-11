#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSquaredKernel
import pandas as pd
import h5py
import os

orange = '#FF9933'
lightblue = '#66CCCC'
blue = '#0066CC'
pink = '#FF33CC'
turquoise = '#3399FF'
lightgreen = '#99CC99'
green = '#009933'
maroon = '#CC0066'
purple = '#9933FF'
red = '#CC0000'
lilac = '#CC99FF'

__all__ = ["teff2bv", "bn_age", "load_data", "plot_2d_prediction",
           "age_model"]

DATA_DIR = "data"

plotpar = {'axes.labelsize': 20,
           'font.size': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 15,
           'ytick.labelsize': 15,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def teff2bv(teff, logg, feh):
    # best fit parameters
    t = [-813.3175, 684.4585, -189.923, 17.40875]
    f = [1.2136, 0.0209]
    d1 = -0.294
    g1 = -1.166
    e1 = 0.3125
    return t[0] + t[1] * np.log10(teff) + t[2] * (np.log10(teff))**2 + \
        t[3] * (np.log10(teff))**3 + f[0] * feh + f[1] * feh**2 + d1 * feh * \
        np.log10(teff) + g1 * logg + e1 * logg * np.log10(teff)


def bn_age(period, bv):
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


def load_data():

    # load clusters
    bv1, bv_err1, p, p_err, a, a_err, logg, logg_err, f = \
        np.genfromtxt(os.path.join(DATA_DIR, "clusters.txt")).T

    # load precise astero
    data = pd.read_csv(os.path.join(DATA_DIR, "vansaders.txt"))
    bv2 = teff2bv(data["Teff"], data["AMP_logg"], data["FeH"])
    bv_err2 = np.ones_like(bv2)*.1

    df = pd.read_csv(os.path.join(DATA_DIR, "chaplin_garcia.csv"))
    dfbvs = teff2bv(df["teff"], df["logg"], df["feh"])

    # load all other astero stars and concatenate
    age = np.concatenate((a, np.array(data["AMP_age"]), df["age"]))
    age_err = np.concatenate((a_err, np.array(data["AMP_age_err"]),
                              .5*(df["age_errp"] + df["age_errm"])))
    period = np.concatenate((p, np.array(data["period"]), df["period"]))
    period_err = period * .1
    bv = np.concatenate((bv1, bv2, dfbvs))
    bv_err = np.concatenate((bv_err1, bv_err2, np.ones_like(dfbvs)*.1))
    logg = np.concatenate((np.ones_like(bv1)*4.4, data["AMP_logg"],
                           df["logg"]))
    feh = np.concatenate((np.ones_like(bv1)*0., data["FeH"], df["feh"]))

    # remove very red stars
    m = (.4 < bv) * (bv < 1.2) * (logg > 4.)
    return age[m], age_err[m], bv[m], bv_err[m], period[m], period_err[m], \
        logg[m], feh[m]


def plot_2d_prediction(id, theta, x, y, z, xerr, yerr, myx, myz, mu, v, xaxis,
                       RESULTS_DIR):

    plt.clf()
    if xaxis == "period":
        bv, period = myz, x
        xplot = np.linspace(0, max(x), 100)
        model_xplot_shape = bn_age(xplot, bv)
        l = theta[2]
        xvar, zvar, xlabel, zlabel = "period", "B-V", \
            "$\mathrm{Period~(Days)}$", "$B-V$"
        m = (myz - .03 < z) * (z < myz + .03)
        plt.xlim(0, max(x[m]) + 5)
    else:
        bv, period = x, myz
        xplot = np.linspace(min(x), max(x), 100)
        model_xplot_shape = bn_age(period, xplot)
        l = theta[1]
        xvar, zvar, xlabel, zlabel = "B-V", "period", "$B-V$", \
            "$\mathrm{Period~(Days)}$"
        m = (myz - (myz * .1 - 10) < z) * (z < (myz * .1 + 10) + myz)
    yvar, ylabel = "age", "$\mathrm{Age~(Gyr)}$"

    plt.errorbar(x[m], y[m], xerr=xerr[m], yerr=yerr[m], fmt="k.", capsize=0,
                 ecolor=".7")
    plt.errorbar(myx, mu, yerr=v, fmt=".", color=pink, ms=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    k = theta[0]**2 * ExpSquaredKernel(l**2)
    gp = george.GP(k)

    model_y_shape = bn_age(period, bv)
    gp.compute(x, (yerr**2 + theta[3]**2)**.5)
    mus, covs = gp.predict(y - model_y_shape, xplot)  # do the prediction
    vs = np.diag(covs)**.5
    plt.plot(xplot, mus + model_xplot_shape, color=blue,
             label=("{0} = {1:.2}".format(zlabel, myz)))
    plt.fill_between(xplot, mus + model_xplot_shape - vs - theta[3],
                     mus + model_xplot_shape + vs + theta[3], alpha=.2,
                     color=blue)
    plt.legend(loc="best")
    plt.savefig(os.path.join(RESULTS_DIR, "{0}_{1}_vs_{2}".format(id, xvar,
                                                                  yvar)))


def age_model(id, mybv, myperiod, RESULTS_DIR, plot=False,):
    """
    Given a rotation period and a colour, calculate the age.
    PARAMETERS:
    mybv: float
        The B-V colour.
    myperiod: float
        The rotation period in days.
    RESULTS_DIR: str
        The path for saving result plot.
    plot: bool
        Makes plot of 2-D projections if true.
    RETURNS:
        The age and uncertainty in Gyr.
    """

    # load the data for conditioning the GP...
    age, age_err, bv, bv_err, period, period_err, logg, feh = load_data()

    # load the results
    with h5py.File(os.path.join(DATA_DIR, "3Dsamples.h5"), "r") as f:
            samples = f["samples"][...]
    nwalkers, nsteps, ndims = np.shape(samples)
    flat = np.reshape(samples, (nwalkers * nsteps, ndims))
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    theta = np.exp(np.array([i[0] for i in mcmc_result]))

    # Calculate the age.
    D = np.vstack((bv, period)).T
    k = theta[0]**2 * ExpSquaredKernel([theta[1]**2, theta[2]**2], ndim=2)
    gp = george.GP(k)
    gp.compute(D, (age_err**2 + theta[3]**2)**.5)
    xs = np.zeros((1, 2))
    xs[0, 0] = mybv
    xs[0, 1] = myperiod
    mu, cov = gp.predict(age-bn_age(period, bv), xs)  # do the GP prediction
    v = np.diag(cov)**.5
    mu += bn_age(myperiod, mybv)

    if plot:
        print("Generating FGKcupid plots...")
        m = (mybv - .1 < bv) * (bv < mybv + .1)
        plot_2d_prediction(id, theta, period[m], age[m], bv[m], period_err[m],
                           age_err[m], myperiod, mybv, mu, v, "period",
                           RESULTS_DIR)

        m = (myperiod - 10 < period) * (period < myperiod + 10)
        plot_2d_prediction(id, theta, bv[m], age[m], period[m], bv_err[m],
                           age_err[m], mybv, myperiod, mu, v, "bv",
                           RESULTS_DIR)

    print("Age = {0} +/- {1} Gyr".format(mu[0], v[0]+theta[3]))
    return mu[0], v[0] + theta[3]


if __name__ == "__main__":
    age_model("test", .65, 26.5, plot=True)

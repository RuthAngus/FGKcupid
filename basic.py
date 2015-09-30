import numpy as np
import matplotlib.pyplot as plt
# import astero as a
import george
from george.kernels import ExpSquaredKernel

def format_data():

    # load asteroseismic data
    data = a.astero()
    m = data.bteff > 0
    KID = data.bKID[m]
    teff, t_err = data.bteff[m], data.bteff_err[m]
    logg, l_err = data.blogg[m], .5*(data.blogg_errp[m] + data.blogg_errm[m])
    feh, f_err = data.bfeh[m], data.bfeh_err[m]
    age, a_err = data.bage[m], .5*(data.bage_errp[m] + data.bage_errm[m])

    # load rotation period data
    data = np.genfromtxt("garcia.txt").T

    # match astero and rotation data
    periods, p_err = [], []
    KIDs = []
    t, terr, l, lerr, f, ferr, ag, agerr = [], [], [], [], [], [], [], []
    for i, kid in enumerate(KID):
        m = kid == data[0]
        if len (data[0][m]):
            KIDs.append(kid)
            periods.append(data[1][m][0])
            p_err.append(data[2][m][0])
            t.append(teff[i])
            terr.append(t_err[i])
            l.append(logg[i])
            lerr.append(l_err[i])
            f.append(feh[i])
            ferr.append(f_err[i])
            ag.append(age[i])
            agerr.append(a_err[i])

    # save results. FIXME: these need to be arrays
    data = np.vstack((KID, p, perr, t, terr, l, lerr, f, ferr, a, aerr))
    np.savetxt("data.txt", data.T)

    return np.array(KIDs), np.array(periods), np.array(p_err), np.array(t), \
            np.array(terr), np.array(l), np.array(lerr), np.array(f), \
            np.array(ferr), np.array(ag), np.array(agerr)

def make_prediction(period, teff):
    """
    Given the training set and a new rotation period, predict an age
    """
    # load data
    KID, p, perr, t, terr, l, lerr, f, ferr, a, aerr = \
            np.genfromtxt("data.txt").T

    # set your x, y z and hyperparameters
    x, y, z, xerr, yerr, zerr = p, a, t, perr, aerr, terr
    D = np.vstack((x, z)).T
    A, lx, lz = 10, 50, 600

    # GP prediction
    k = A**2 * ExpSquaredKernel([lx**2, lz**2], ndim=2)
    gp = george.GP(k)
    gp.compute(D, yerr)

    xs = np.zeros((1, 2))
    xs[0, 0] = period
    xs[0, 1] = teff

    mu, cov = gp.predict(y, xs)  # do the prediction
    v = np.diag(cov)**.5
    return mu[0], v[0]

def make_prediction_1D():
    """
    Given the training set and a new rotation period, predict an age
    """
    # load data
    KID, p, perr, t, terr, l, lerr, f, ferr, a, aerr = \
            np.genfromtxt("data.txt").T

    # set your x, y and hyperparameters
    x, y, xerr, yerr = l, a, lerr, aerr
    A, l = 10, .8

    # GP prediction
    k = A**2 * ExpSquaredKernel(l**2)
    gp = george.GP(k)
    gp.compute(x, yerr)

    xs = np.linspace(min(x), max(x), 1000)

    mu, cov = gp.predict(y, xs)  # do the prediction
    v = np.diag(cov)**.5

    plt.clf()
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.")
    plt.plot(xs, mu)
    plt.savefig("test")

if __name__ == "__main__":

    make_prediction_1D()
#     p, perr = make_prediction(26, 5700)
#     print p, perr

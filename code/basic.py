import numpy as np
import matplotlib.pyplot as plt
import astero
import george
from george.kernels import ExpSquaredKernel

def format_data():

#     # load asteroseismic data
#     data = a.astero()
#     m = data.bteff > 0
#     KID = data.bKID[m]
#     teff, t_err = data.bteff[m], data.bteff_err[m]
#     logg, l_err = data.blogg[m], .5*(data.blogg_errp[m] + data.blogg_errm[m])
#     feh, f_err = data.bfeh[m], data.bfeh_err[m]
#     age, a_err = data.bage[m], .5*(data.bage_errp[m] + data.bage_errm[m])

    data = astero.astero()
    m = data.iteff > 0
    KID = data.iKID[m]
    teff, t_err = data.iteff[m], data.iteff_err[m]
    logg, l_err = data.ilogg[m], .5*(data.ilogg_errp[m] + data.ilogg_errm[m])
    feh, f_err = data.ifeh[m], data.ifeh_err[m]
    age, a_err = data.iage[m], .5*(data.iage_errp[m] + data.iage_errm[m])

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
    KIDs, p, perr, t, terr, l, lerr, f, ferr, a, aerr = np.array(KIDs), \
            np.array(periods), np.array(p_err), np.array(t), np.array(terr), \
            np.array(l), np.array(lerr), np.array(f), np.array(ferr), \
            np.array(ag), np.array(agerr)
    data = np.vstack((KIDs, p, perr, t, terr, l, lerr, f, ferr, a, aerr))
#     np.savetxt("data.txt", data.T)
    np.savetxt("big_data.txt", data.T)

#     return np.array(KIDs), np.array(periods), np.array(p_err), np.array(t), \
#             np.array(terr), np.array(l, np.array(lerr), np.array(f), \
#             np.array(ferr), np.array(ag), np.array(agerr)

def make_prediction(period, teff, logg, feh):
    """
    Given the training set and a new rotation period, predict an age
    """

    # load data
    KID, p, perr, t, terr, l, lerr, f, ferr, a, aerr = \
            np.genfromtxt("big_data.txt").T
    m = (t < 6250) * (l > 4.2) * (p > 1)
    KID, p, perr, t, terr, l, lerr, f, ferr, a, aerr = \
            KID[m], p[m], perr[m], t[m], terr[m], l[m], lerr[m], f[m], \
            ferr[m], a[m], aerr[m]

    # set your x, y z and hyperparameters
    x, y, z, u, w, xerr, yerr, zerr, uerr, werr = \
            p, a, t, l, f, perr, aerr, terr, lerr, ferr
    D = np.vstack((x, z, u, w)).T
    A, lx, lz, lu, lw = 10, 50, 600, .8, .3

    # GP prediction
    k = A**2 * ExpSquaredKernel([lx**2, lz**2, lu**2, lw**2], ndim=4)
    gp = george.GP(k)
    gp.compute(D, yerr)

    xs = np.zeros((1, 4))
    xs[0, 0] = period
    xs[0, 1] = teff
    xs[0, 2] = logg
    xs[0, 3] = feh

    mu, cov = gp.predict(y, xs)  # do the prediction
    v = np.diag(cov)**.5
    return mu[0], v[0]

def make_prediction_1D():
    """
    Given the training set and a new rotation period, predict an age
    """
    # load data
    KID, p, perr, t, terr, l, lerr, f, ferr, a, aerr = \
            np.genfromtxt("big_data.txt").T
#             np.genfromtxt("data.txt").T

    # set your x, y and hyperparameters
    x, y, xerr, yerr = f, a, ferr, aerr
    A, l = 10, .3

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

#     format_data()
#     assert 0
#     make_prediction_1D()
#     assert 0
    p, perr = make_prediction(26, 5700, 4.4, 0)
    print p, perr

import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
import corner
import os

lightblue = '#66CCCC'

plotpar = {'axes.labelsize': 20,
           'font.size': 20,
           'legend.fontsize': 20,
           'xtick.labelsize': 15,
           'ytick.labelsize': 15,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def make_plot(sampler, x, y, yerr, fig_labels, DIR, traces=False, tri=False,
              prediction=True):

    nwalkers, nsteps, ndims = np.shape(sampler)
    flat = np.reshape(sampler, (nwalkers * nsteps, ndims))
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    mcmc_result = np.array([i[0] for i in mcmc_result])
    print("\n", np.exp(np.array(mcmc_result[-1])), "period (days)", "\n")
    print(mcmc_result)
    np.savetxt(os.path.join(DIR, "result.txt"), mcmc_result)

    if traces:
        print("Plotting traces")
        for i in range(ndims):
            plt.clf()
            plt.plot(sampler[:, :, i].T, 'k-', alpha=0.3)
            plt.ylabel(fig_labels[i])
            plt.savefig(os.path.join(DIR, "{0}".format(fig_labels[i])))

    if tri:
        print("Making triangle plot")
        flat[:, -1] = np.exp(flat[:, -1])
        fig = corner.corner(flat, labels=fig_labels)
        fig.savefig(os.path.join(DIR, "triangle"))

    if prediction:
        print("plotting prediction")
        theta = np.exp(np.array(mcmc_result))
        k = theta[0] * ExpSquaredKernel(theta[1]) + WhiteKernel(theta[2])
        gp = george.GP(k, solver=george.HODLRSolver)
        gp.compute(x-x[0], yerr)
        xs = np.linspace((x-x[0])[0], (x-x[0])[-1], 1000)
        mu, cov = gp.predict(y, xs)
        v = np.diag(cov)**.5
        plt.clf()
        plt.errorbar(x-x[0], y, yerr=yerr, fmt="k.", capsize=0)
        plt.xlabel("$\mathrm{Time~(days)}$")
        plt.ylabel("$\mathrm{Normalised~Flux}$")
        plt.plot(xs, mu, color=lightblue)
        plt.fill_between(xs, mu-v, mu+v, color=lightblue, alpha=.2)
        plt.xlim(min(x-x[0]), max(x-x[0]))
        plt.savefig(os.path.join(DIR, "prediction"))

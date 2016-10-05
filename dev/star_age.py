#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones import StarModel
import pandas as pd
import os
import kplr
import glob
from fgkcupid import teff2bv, age_model
from kepler_data import load_kepler_data
import acf

DATA_DIR = "data"
LC_DIR = "/Users/ruthangus/.kplr/data/lightcurves"
RESULTS_DIR = "results"


class star(object):

    def __init__(self, id, mass=None, teff=None, logg=None, feh=None,
                 prot=None, BV=None, Vmag=None, Gmag=None, kepmag=None,
                 parallax=None, DATA_DIR=DATA_DIR, LC_DIR=LC_DIR,
                 download_lc=True):
        """
        Routines for calculating the age of a single star. Currently only
        suitable for Kepler targets.
        PARAMETERS
        ----------
        id: str
            The id of the star. A kepler star will have a 9-digit integer.
        mass: tuple (mass, mass_err) (optional)
            Mass in Solar masses.
        teff: tuple (teff, teff_err) (optional)
            Effective temperature (K).
        logg: tuple (logg, logg_err) (optional)
            log (g).
        feh: tuple (feh, feh_err) (optional)
            Iron abundance.
        prot: tuple (prot, prot_err) (optional)
            The rotation period in days.
        BV: tuple (B-V, B-V_err) (optional)
            The B-V colour.
        Vmag: tuple (Vmag, Vmag_err) (optional)
            The V-band magnitude. This is currently just a placeholder.
        Gmag: tuple (Gmag, Gmag_err) (optional)
            The Gaia magnitude.
        kepmag: tuple (kepmag, kepmag_err) (optional)
            The Kepler magnitude.
        parallax: tuple (parallax, parallax_err) (optional)
            The Gaia parallax.
        DATA_DIR: str
            The directory containing training data for the age model.
        LC_DIR: str
            The directory containing the kepler light curves.
        download_lc: bool
            if True the Kepler light curve is downloaded and an ACF computed.
        """

        assert len(id) < 10, "ID must be a 9-digit KIC id."
        self.id = str(int(id)).zfill(9)  # make sure id is in KIC format.

        # KIC parameters.
        if not teff or logg or feh or kepmag:
            client = kplr.API()
            star = client.star(self.id)
            if not teff:
                self.teff = (star.kic_teff, 50)
            if not logg:
                self.logg = (star.kic_logg, .1)
            if not feh:
                self.feh = (star.kic_feh, .1)
            if not kepmag:
                self.kepmag = (star.kic_kepmag, .1)
        if not BV:
            self.BV = (teff2bv(self.teff[0], self.logg[0], self.feh[0]), .01)

        # Load Gaia parameters.
        if not parallax:
            Gdata = pd.read_csv(os.path.join(DATA_DIR,
                                             "ruth_matched.csv"))
            m = np.array(Gdata["kepid"]) == int(self.id)
            self.parallax = (float(np.array(Gdata["parallax"])[m]),
                             float(np.array(Gdata["parallax_error"])[m]))

        # load rotation period.
        if not prot:
            # Search McQuillan catalogue.
            Rdata = pd.read_csv(os.path.join(DATA_DIR,
                                             "Table_1_Periodic.txt"))
            m = np.array(Rdata["KID"]) == int(self.id)
            if len(Rdata["KID"][m]):
                self.prot = (float(np.array(Rdata["Prot"])[m]),
                             float(np.array(Rdata["Prot_err"])[m]))
                print("Rotation period from McQuillan et al. (2013)")
            else:
                # Search Garcia catalogue.
                Rdata = pd.read_csv(os.path.join(DATA_DIR,
                                                 "chaplin_garcia.csv"))
                m = np.array(Rdata["kid"]) == int(self.id)
                if len(Rdata["kid"][m]):
                    self.prot = (float(np.array(Rdata["period"])[m]),
                                 float(np.array(Rdata["period_err"])[m]))
                    print("Rotation period from Garcia et al. (2014)")
                else:
                    print("No period found")

        # If no rotation period exists download the light curve and
        # calculate acf.
        # load or download light curve.
        result_file = os.path.join(RESULTS_DIR,
                                   "{}_result.txt".format(self.id))
        if os.path.exists(result_file):
            print("Previous rotation period measurement found")
            prot = tuple(np.genfromtxt(result_file))
        else:
            if not prot:
                fnames = glob.glob(os.path.join(LC_DIR,
                                                "{}/*fits".format(self.id)))
                if not len(fnames):
                    print("Downloading light curve...")
                    client = kplr.API()
                    star = client.star(self.id)
                    star.get_light_curves(fetch=True, shortcadence=False)
                fnames = glob.glob(os.path.join(LC_DIR,
                                                "{}/*fits".format(self.id)))
                x, y, yerr = load_kepler_data(fnames)

                # calculate ACF
                self.prot = acf.corr_run(x, y, yerr, self.id,
                                         savedir="results",
                                         saveplot=True)

        # Calculate a gyrochronology age using fgkcupid.
        fname = os.path.join(RESULTS_DIR, "{}_gyro_age.txt".format(self.id))
        if os.path.exists(fname):
            self.gyro_age = tuple(np.genfromtxt(fname))
        else:
            print("Calculating rotation-age")
            self.gyro_age = age_model(self.BV[0], self.prot[0], plot=True)
            np.savetxt(fname, self.gyro_age)

    def isochronal_age(self):
        """
        Calculate the isochronal age of the star using Tim Morton's isochrones
        code.
        """

        fname = os.path.join(RESULTS_DIR, "{}.h5".format(self.id))
        if os.path.exists(fname):
            mod = StarModel.load_hdf(fname)
        else:

            # set maxage
            maxage = self.gyro_age[0] + self.gyro_age[1]

            # set up StarModel
            dar = Dartmouth_Isochrone()
            mod = StarModel(dar, Teff=self.teff, logg=self.logg, feh=self.feh,
                            Kepler=self.kepmag, maxage=maxage, use_emcee=True)

            # Calculating isochronal age. Running MCMC..."
            mod.fit_mcmc()
            mod.save_hdf(fname)

        age_samples = mod.prop_samples("age")
        age, p1, p2 = np.percentile(age_samples[0], [16, 50, 84])
        age_errp, age_errm = p2 - age, age - p1
        return age, age_errp, age_errm, age_samples


def match(id1, id2):
    """
    Find the common ids between two lists of ids.
    id1: array
        Small id list.
    id2: array
        Large id list.
    Returns matched ids and indices of matched arrays.
    """
    matched = []
    inds1, inds2 = [], []
    for i, id in enumerate(id1):
        m = id2 == id
        if len(id2[m]):
            matched.append(id)
            inds2.append(int(np.where(m)[0]))
            inds1.append(i)
    return matched, inds1, inds2


if __name__ == "__main__":
    st = star("002450729")
    st.isochronal_age()

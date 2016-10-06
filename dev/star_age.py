#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
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
                 parallax=None, gyro_age=None, DATA_DIR=DATA_DIR,
                 LC_DIR=LC_DIR, download_lc=True):
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
        parallax: tuple (Gyro_age, gyro_age_err) (optional)
            The gyro age.
        DATA_DIR: str
            The directory containing training data for the age model.
        LC_DIR: str
            The directory containing the kepler light curves.
        download_lc: bool
            if True the Kepler light curve is downloaded and an ACF computed.
        """

        assert len(id) < 10, "ID must be a 9-digit KIC id."
        self.id = str(int(id)).zfill(9)  # make sure id is in KIC format.
        self.mass, self.teff, self.logg, self.feh = mass, teff, logg, feh
        self.prot, self.BV, self.Vmag, self.Gmag = prot, BV, Vmag, Gmag
        self.kepmag, self.parallax, self.gyro_age = kepmag, parallax, gyro_age

        # KIC parameters.
        if not teff or logg or feh or kepmag:
            client = kplr.API()
            kic_star = client.star(self.id)
            if not self.teff:
                self.teff = (kic_star.kic_teff, 50)
            if not self.logg:
                self.logg = (kic_star.kic_logg, .1)
            if not self.feh:
                self.feh = (kic_star.kic_feh, .1)
            if not self.kepmag:
                self.kepmag = (kic_star.kic_kepmag, .1)

        # If the KIC values are empty, search in the astero database.
        if not self.teff[0] or not self.logg[0] or not self.feh[0]:
            d = pd.read_csv(os.path.join(DATA_DIR, "vansaders.txt"))
            m = np.array(d["KIC"]) == str(int(self.id))
            if len(np.array(d["KIC"])[m]):
                print("Loading value from van Saders et al. (2016)")
                self.teff = (float(np.array(d["Teff"])[m]),
                             float(np.array(d["Teff_err"])[m]))
                self.logg = (float(np.array(d["AMP_logg"])[m]), .1)
                self.feh = (float(np.array(d["FeH"])[m]), .01)
                print(self.teff, self.logg, self.feh)
            else:
                import astero
                ast = astero.astero()
                m = ast.iKID == float(self.id)
                if len(ast.iKID[m]):
                    print("Loading value from Chaplin et al. (2014)")
                    self.teff = (float(ast.iteff[m]), float(ast.iteff_err[m]))
                    self.logg = (float(ast.ilogg[m]),
                                 (float(ast.ilogg_errm[m]) +
                                  float(ast.ilogg_errp[m])*.5))
                    self.feh = (-.2, .01)

        # Convert teff, logg and feh to B-V colour.
        if not self.BV:
            self.BV = (teff2bv(self.teff[0], self.logg[0], self.feh[0]), .01)

        # Load Gaia parameters.
        if not self.parallax:
            Gdata = pd.read_csv(os.path.join(DATA_DIR,
                                             "ruth_matched.csv"))
            m = np.array(Gdata["kepid"]) == int(self.id)
            try:
                self.parallax = (float(np.array(Gdata["parallax"])[m]),
                                 float(np.array(Gdata["parallax_error"])[m]))
                print("Gaia parallax found")
            except:
                print("Object not found in the TGAS catalogue.")

        # load rotation period.
        if not self.prot:
            # Search van saders catalogue.
            d = pd.read_csv(os.path.join(DATA_DIR, "vansaders.txt"))
            m = np.array(d["KIC"]) == str(int(self.id))
            if len(np.array(d["KIC"][m])):
                self.prot = (float(np.array(d["period"])[m]),
                             float(np.array(d["period_err"])[m]))
                print("Rotation period from van Saders (2016)")
            else:
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
        if not self.prot:
            result_file = os.path.join(RESULTS_DIR,
                                       "{}_result.txt".format(self.id))
            if os.path.exists(result_file):
                print("Previous rotation period measurement found")
                self.prot = tuple(np.genfromtxt(result_file))
            else:
                fnames = glob.glob(os.path.join(LC_DIR,
                                                "{}/*fits".format(self.id)))
                if not len(fnames):
                    print("Downloading light curve...")
                    client = kplr.API()
                    star = client.star(self.id)
                    star.get_light_curves(fetch=True, shortcadence=False)
                    fnames = \
                        glob.glob(os.path.join(LC_DIR,
                                               "{}/*fits".format(self.id)))
                    x, y, yerr = load_kepler_data(fnames)

                    # calculate ACF
                    self.prot = acf.corr_run(x, y, yerr, self.id,
                                             savedir="results",
                                             saveplot=True)

        if self.prot:
            # Calculate a gyrochronology age using fgkcupid.
            fname = os.path.join(RESULTS_DIR,
                                 "{}_gyro_age.txt".format(self.id))
            if os.path.exists(fname):
                self.gyro_age = tuple(np.genfromtxt(fname))
            else:
                print("Calculating rotation-age")
                self.gyro_age = age_model(self.BV[0], self.prot[0], plot=True)
                np.savetxt(fname, self.gyro_age)
            print("Gyro age = {0:.2} +/- {1:.2} Gyr".format(self.gyro_age[0],
                                                            self.gyro_age[1]))

    def isochronal_age(self, gyro_prior=False, use_Gaia_parallax=True):
        """
        Calculate the isochronal age of the star using Tim Morton's isochrones
        code.
        """
        if not use_Gaia_parallax:
            self.parallax = None

        if gyro_prior:
            if not self.gyro_age:
                print("A rotation period is needed to add a gyro age prior \
                      --- none found.")
            gyro_prior = False

        fname = os.path.join(RESULTS_DIR, "{}.h5".format(self.id))
        if os.path.exists(fname):
            mod = StarModel.load_hdf(fname)
        else:
            # set maxage
            # maxage = self.gyro_age[0] + self.gyro_age[1]

            # set up StarModel
            dar = Dartmouth_Isochrone()
            mod = StarModel(dar, Teff=self.teff, logg=self.logg, feh=self.feh,
                            Kepler=self.kepmag, parallax=self.parallax,
                            use_emcee=True)

            print("Calculating isochronal age. Running MCMC...")
            mod.fit_mcmc()
            mod.save_hdf(fname)

        age_samples = mod.prop_samples("age")
        age, p1, p2 = np.percentile(age_samples[0], [16, 50, 84])
        age_errp, age_errm = p2 - age, age - p1
        return age, age_errp, age_errm, age_samples, mod


if __name__ == "__main__":
    st = star("6196457")
    # st = star("002450729")
    st.isochronal_age()

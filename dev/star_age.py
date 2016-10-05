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
from fgkcupid import teff2bv
from kepler_data import load_data

DATA_DIR = "data"
LC_DIR = "/Users/ruthangus/.kplr/data/lightcurves"


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
        id = str(int(id)).zfill(9)  # make sure id is in KIC format.

        # KIC parameters.
        if not teff or logg or feh or kepmag:
            client = kplr.API()
            star = client.star(id)
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
            m = np.array(Gdata["kepid"]) == int(id)
            self.parallax = (float(np.array(Gdata["parallax"])[m]),
                             float(np.array(Gdata["parallax_error"])[m]))

        # load rotation period.
        if not prot:
            # Search McQuillan catalogue.
            Rdata = pd.read_csv(os.path.join(DATA_DIR,
                                             "Table_1_Periodic.txt"))
            m = np.array(Rdata["KID"]) == int(id)
            if len(Rdata["KID"][m]):
                self.prot = (float(np.array(Rdata["Prot"])[m]),
                             float(np.array(Rdata["Prot_err"])[m]))
                print("Rotation period from McQuillan et al. (2013)")
            else:
                # Search Garcia catalogue.
                Rdata = pd.read_csv(os.path.join(DATA_DIR,
                                                 "chaplin_garcia.csv"))
                m = np.array(Rdata["kid"]) == int(id)
                if len(Rdata["kid"][m]):
                    self.prot = (float(np.array(Rdata["period"])[m]),
                                 float(np.array(Rdata["period_err"])[m]))
                    print("Rotation period from Garcia et al. (2014)")
                else:
                    print("No period found")

        # If no rotation period exists download the light curve and
        # calculate acf.
        # load or download light curve.
        if not prot:
            fnames = glob.glob(os.path.join(LC_DIR, "{}/*fits".format(id)))
            if not len(fnames):
                client = kplr.API()
                star = client.star(id)
                star.get_light_curves(fetch=True, shortcadence=False)
            fnames = glob.glob(os.path.join(LC_DIR, "{}/*fits".format(id)))

        # calculate ACF




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
    star("002450729")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones import StarModel
import pandas as pd
import os
import kplr
import fgkcupid as fc

DATA_DIR = "data"
LC_DIR = "/Users/ruthangus/.kplr/data/lightcurves"


class star(object):

    def __init__(self, id, mass=None, teff=None, logg=None, feh=None,
                 prot=None, BV=None, Vmag=None, Gmag=None, kepmag=None,
                 parallax=None, DATA_DIR=DATA_DIR, LC_DIR=LC_DIR):
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
        """

        assert len(id) < 10, "ID must be a 9-digit KIC id."
        id = str(int(id)).zfill(9)  # make sure id is in KIC format.

        # KIC parameters.
        if not teff or logg or feh or kepmag:
            client = kplr.API()
            star = client.star(id)
            if not teff:
                teff = (star.kic_teff, 50)
            if not logg:
                logg = (star.kic_logg, .1)
            if not feh:
                feh = (star.kic_feh, .1)
            if not kepmag:
                kepmag = (star.kic_kepmag, .1)
        if not BV:
            BV = (fc.teff2bv(teff, logg, feh), .01)

        # Gaia parameters.
        if not parallax:
            Gdata = pd.read_csv(os.path.join(DATA_DIR,
                                             "ruth_matched.csv"))
            m = Gdata["kepid"] == id
            parallax = (Gdata["parallax"][m], Gdata["parallax_error"][m])

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

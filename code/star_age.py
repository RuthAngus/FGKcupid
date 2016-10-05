#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones import StarModel
import pandas as pd
import os
import kplr

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
            The V-band magnitude.
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

        # initialise kplr if any KIC values are missing.
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
        print(teff, logg, kepmag)


if __name__ == "__main__":
    star("002450729")

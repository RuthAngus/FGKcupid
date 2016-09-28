# Code for downloading the EPIC catalogue
from __future__ import division, print_function
import os
import requests
import pandas as pd
import numpy as np

def get_catalog(name, basepath="data"):
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name)
    csvfn = os.path.join(basepath, "{0}.csv".format(name))
    if not os.path.exists(csvfn):
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        print("Downloading {0}...".format(name))
        url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
               "nph-nstedAPI?table={0}&select=*").format(name)
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            r.raise_for_status()
        with open(csvfn, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(csvfn, low_memory=False)
    df.to_hdf(fn, name, format="t")
    return df

if __name__ == "__main__":
    df = get_catalog("k2targets")

    ids = df.epic_number
    inds = np.argsort(df.epic_number)
    print(df.keys())
    teffs = df.k2_teff
    b = df.k2_bjmag
    v = df.k2_vjmag
    j = df.k2_jmag
    k = df.k2_kmag
    print(k[inds])
    ra, raerr = df.ra, df.raerr
    dec, decerr = df.dec, df.decerr
    glon, glonerr = df.k2_glon, df.k2_glonerr
    idlist = ids[1951:3626]

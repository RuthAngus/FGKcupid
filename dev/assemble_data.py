import numpy as np
import pandas as pd
import kplr
client = kplr.API()
import os
from download_epic import get_catalog

PATH = "data"


def match(ids1, ids2):
    matched_ids = []
    inds1, inds2 = [], []
    for i, id in enumerate(ids1):
        m = ids2 == id
        if len(ids2[m]):
            matched_ids.append(id)
            inds1.append(i)
            inds2.append(np.where(m)[0][0])
    return matched_ids, inds1, inds2


def K2_cluster(name, ids, periods, age, age_err, feh, feh_err):
    """
    Using measurements of rotation period from the literature, gather
    effective temperatures and log (g)s from the EPIC catalogue.
    Save the data in a csv file.
    """

    epic = get_catalog("k2targets")
    matched_ids, inds1, inds2 = match(ids, epic.epic_number)
    print(epic.keys())

    d = {"epic_id": matched_ids,
         "age": np.ones_like(matched_ids)*age,
         "age_err": np.ones_like(matched_ids)*age_err,
         "feh": np.ones_like(matched_ids)*feh,
         "feh_err": np.ones_like(matched_ids)*feh_err,
         "logg": epic.k2_logg[inds2],
         "logg_err": .5 * (epic.k2_loggerr1[inds2] + epic.k2_loggerr2[inds2]),
         "period": periods[inds1],
         "teff": epic.k2_teff[inds2],
         "teff_err": .5 * (epic.k2_tefferr1[inds2] + epic.k2_tefferr2[inds2])}

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(PATH, "{}.csv".format(name)))

if __name__ == "__main__":
    # hyades rotation periods from Douglas et al. (2016)
    hyades_ids, hyades_prots = \
        np.genfromtxt(os.path.join(PATH, "J_ApJ_822_47/table4.txt")).T

    # Hyades age from Perryman et al. (1998)
    K2_cluster("hyades", hyades_ids, hyades_prots, .625, .05, .17, 0.)

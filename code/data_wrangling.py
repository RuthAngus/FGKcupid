import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt("metcalfe.txt", skip_header=1).T
data2 = np.genfromtxt("garcia.txt").T

# match kids
kids, mass, masserr, age, aerr = [], [], [], [], []
z, zerr = [], []
p, perr, t, terr = [], [], [], []
for i, kid in enumerate(data1[0]):
    m = kid == data2[0]
    if len(data2[0][m]):
        kids.append(data2[0][m][0])
        mass.append(data1[3][i])
        masserr.append(data1[4][i])
        age.append(data1[5][i])
        aerr.append(data1[6][i])
        z.append(data1[7][i])
        zerr.append(data1[8][i])
        p.append(data2[1][m][0])
        perr.append(data2[2][m][0])
        t.append(data2[3][m][0])
        terr.append(data2[4][m][0])

data = np.vstack((np.array(kids), np.array(mass), np.array(masserr),
                 np.array(age), np.array(aerr), np.array(z), np.array(zerr),
                 np.array(p), np.array(perr), np.array(t), np.array(terr)))

np.savetxt("metcalfe_sample.txt", data.T)

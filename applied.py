import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

from linear import nu, step
from stationary import F, root

def localize(M, T):
    s = np.zeros(3)
    while not any(s):
        # print('Minimizing')
        # x0 = source + np.random.normal(0, 0.1, N)
        s0 = np.random.normal(0, 0.25, 3)
        # x0 = np.zeros(N)
        res = root(F, s0, args=(M, T), method='lm')
        s = np.array([np.sign(eks)*min(3, abs(eks)) for eks in res.x])

    return s

M = read_csv('./data/microphones.csv', header=None).to_numpy()
toa_raw = read_csv('./data/toa_measurements.csv', header=None)
toa_clean = toa_raw[~toa_raw.isna().any(axis=1)]
indices = toa_clean.index.to_numpy()
TOA = toa_raw.to_numpy()

times = read_csv('./data/times.csv', header=None).to_numpy()[0]
# times = times[indices]

sound_path = read_csv('./data/sound_path.csv', header=None).to_numpy()

alpha = 1e-1
sigma = 1e-2

t0 = 0
x = np.zeros(6)
P = np.eye(6)*sigma**2

S = []
path = []
for k, t in enumerate(times):
    if k % 1000 == 0:
        print(k)
    dt = t - t0
    rel = np.argwhere(~np.isnan(TOA[k]))
    rel = np.reshape(rel, (len(rel),))
    M_rel = M[rel,:]    # Relevant mics
    T_rel = TOA[k,rel]
    # print(M_rel)
    # print(T_rel)

    if len(T_rel) > 11: # Threshold for amount of active mics
        s = sound_path[k,:]
        if len(s[s == s]) == 3:
            z = localize(M_rel, T_rel)
            x, P = step(x, P, M_rel, z, dt, alpha, sigma)
        
            S.append(s)
            path.append(x[:3])

    t0 = t

path = np.array(path)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(M[:,0], M[:,1], M[:,2], color='green', marker='x')
ax.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], 'r--')
ax.plot([s[0] for s in S], [s[1] for s in S], [s[2] for s in S], color='blue')
plt.show()

if __name__ == '__main__':
    pass
    # print(toa_clean)
    # print(indices)
    # print(times)
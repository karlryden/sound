import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

nu = 340

def setup(N, n):
    M = np.random.rand(n, N) - 0.5*np.ones((n, N))
    source = np.random.normal(0, 0.125, N)
    mag = np.linalg.norm(M - source, axis=1)
    e = np.array([np.random.normal(0, a*10e-5) for a in mag])
    T = mag/nu + e

    return M, source, T


def F(s, M, T):
    n, _ = M.shape
    ret = []

    for i in range(n):
        for j in range(i + 1, n):
            m_i = M[i,:]
            m_j = M[j,:]
            d_ij = nu*(T[j] - T[i])

            ret.append(np.linalg.norm(m_j - s) - np.linalg.norm(m_i - s) - d_ij)

    # print(f'ret: {np.array(ret)}')

    return np.array(ret)

def localize(M, T):
    _, N = M.shape
    s = np.zeros(N)
    while (not any(s)) or (np.max(np.abs(s)) > 0.5):
        # print('Minimizing')
        # x0 = source + np.random.normal(0, 0.1, N)
        x0 = np.random.normal(0, 0.125, N)
        res = root(F, x0, args=(M, T), method='lm')
        s = res.x

    return s

def estimate(M, source, num=10):
    mag = np.linalg.norm(M - source, axis=1)
    S = np.ndarray((num, N))

    for i in range(num):
        e = np.array([np.random.normal(0, a*10e-5) for a in mag])
        T = mag/nu + e
        s = localize(M, T)
        S[i,:] = s

    return S

def visualize(M, source, S):
    _, N = S.shape
    est = np.mean(S, axis=0)

    fig = plt.figure()

    if N == 2:
        C = np.cov(S.T)
        w, v = np.linalg.eig(C)
        ax = fig.add_subplot()
        ell = Ellipse(
            xy=est, 
            width=2*math.sqrt(w[0]), 
            height=2*math.sqrt(w[1]), 
            angle=np.rad2deg(np.arctan2(v[1, 1], v[0, 1]) + np.pi/2),
            edgecolor='orange',
            facecolor='red',
            alpha=0.25
        )
        ax.add_patch(ell)

        plt.scatter(M[:,0], M[:,1], color='g', marker='x', linewidths=0.5)
        plt.scatter(source[0], source[1], color='b', linewidths=5)
        plt.scatter(est[0], est[1], color='r', linewidths=3)
        plt.scatter([s[0] for s in S], [s[1] for s in S], color='orange')

        for j in range(2):
            l = w[j]
            e = v[:,j]
            c = math.sqrt(l)*e + est
            plt.plot([est[0], c[0]], [est[1], c[1]], 'r--')

        # plt.legend(['Cov1', 'Cov2', 'Confidence Ellipse', 'Microphones', 'True source', 'Estimated source', 'Guesses'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')

    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(M[:,0], M[:,1], M[:,2], color='g', marker='x')
        ax.scatter(source[0], source[1], source[2], color='b', linewidths=5)
        ax.scatter(est[0], est[1], est[2], color='r', linewidths=3)
        ax.scatter([s[0] for s in S], [s[1] for s in S], [s[2] for s in S], color='orange')

        # ax.legend(['Microphones', 'True source', 'Estimated source'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    else:
        print('Invalid dimension value (N).')
        del fig
        return

    plt.show()

def minscape_surf(M, source, S, T):
    L = 99
    I = np.linspace(-0.5, 0.5, L)
    [X, Y] = np.meshgrid(I, I)

    est = np.mean(S, axis=0)

    Z = np.ndarray((L, L))

    for i in range(L):
        for j in range(L):
            x = I[j]
            y = I[i]
            Z[i,j] = np.linalg.norm(F(np.array([x, y]), M, T))**2

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(M[:,0], M[:,1], [0 for _ in range(n)], color='g', marker='x')
    ax.scatter(source[0], source[1], 0, color='b', linewidths=5)
    ax.scatter([s[0] for s in S], [s[1] for s in S], [0 for _ in S], color='orange')
    ax.plot_wireframe(X, Y, Z, rcount=9, ccount=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('|f(x, y)|^2')

    plt.show()

def minscape_heatmap(M, source, S, T):
    L = 99
    I = np.linspace(-0.5, 0.5, L)

    est = np.mean(S, axis=0)

    Z = np.ndarray((L, L))

    for i in range(L):
        for j in range(L):
            x = I[j]
            y = I[i]
            Z[i,j] = np.linalg.norm(F(np.array([x, y]), M, T))**2

    fig = plt.figure()
    plt.imshow(Z, extent=(-0.5, 0.5, -0.5, 0.5), cmap='binary', alpha=0.5)
    plt.scatter(M[:,0], M[:,1], color='g', marker='x', linewidths=0.5)
    plt.scatter(source[0], source[1], color='b', linewidths=5)
    plt.scatter(est[0], est[1], color='r', linewidths=3)
    plt.scatter([s[0] for s in S], [s[1] for s in S], color='orange')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


if __name__ == '__main__':
    N = 2
    n = 4
    M, source, T = setup(N, n)

    S = estimate(M, source, 100)
    print(f'Estimated source: \n {np.mean(S, axis=0)}')
    print(f'Covariance matrix: \n {np.cov(S.T)}')

    minscape_surf(M, source, S, T)
    minscape_heatmap(M, source, S, T)

    visualize(M, source, S)

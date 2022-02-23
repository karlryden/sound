import numpy as np
import math
import matplotlib.pyplot as plt
from stationary import nu, F, localize, measure

def setup(N, n, sigma):
    M = np.random.rand(n, N) - 0.5*np.ones((n, N))
    source = np.random.normal(0, 0.125, N)
    mag = np.linalg.norm(M - source, axis=1)
    # T = measure(M, source)
    # x0 = np.block([localize(M, T), np.zeros(N)])
    x0 = np.zeros(2*N)
    P0 = np.eye(2*N)*sigma**2

    return M, source, x0, P0


def step(xold, Pold, M, z, dt, alpha, sigma):
    _, N = M.shape
    A = np.block([[np.eye(N),     dt*np.eye(N)],
                  [np.zeros((N, N)), np.eye(N)]])

    B = np.block([[dt**2/2*np.eye(N)],
                  [dt*np.eye(N)     ]])

    C = np.block([np.eye(N), np.zeros((N, N))])

    Q = alpha**2*np.block([[dt**4/4*np.eye(N), dt**3/2*np.eye(N)],
                           [dt**3/2*np.eye(N), dt**2*np.eye(N)  ]])

    R = sigma**2*np.eye(N)

    xmid = np.dot(A, xold)
    # z = localize(M, T) + np.random.normal(0, sigma, N)
    # z = s + np.random.normal(0, sigma, N)
    y = z - np.dot(C, xmid)

    Pmid = np.dot(A, np.dot(Pold, A.T)) + Q
    S = np.dot(C, np.dot(Pmid, C.T)) + R
    K = np.dot(Pmid, np.dot(C.T, np.linalg.inv(S)))
    xnew = xold + np.dot(K, y)
    Pnew = np.dot(np.eye(2*N) - np.dot(K, C), Pmid)

    return xnew, Pnew

def simulate(N, n, dt, alpha, sigma, num=10):
    M, s, x, P = setup(N, n, sigma)
    path = np.ndarray((num, N))
    S = np.ndarray((num, N))

    v = np.zeros(N)

    for i in range(num):
        S[i,:] = s

        T = measure(M, s) #+ np.random.normal(0, sigma, n)
        z = localize(M, T) + np.random.normal(0, sigma, N)
        x, P = step(x, P, M, z, dt, alpha, sigma)

        path[i,:] = x[:N]

        a = np.random.normal(0, alpha, N)
        v += dt*a
        s += dt*v + dt**2/2*a

    return M, S, path


def visualize(M, S, path):
    _, N = M.shape

    fig = plt.figure()

    if N == 2:
        plt.scatter(M[:,0], M[:,1], color='g', marker='x')
        plt.plot([source[0] for source in S], [source[1] for source in S], color='blue')
        plt.plot([p[0] for p in path], [p[1] for p in path], 'r--')
        plt.xlabel('x')
        plt.ylabel('y')

    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(M[:,0], M[:,1], M[:,2], color='g', marker='x')
        ax.plot([source[0] for source in S], [source[1] for source in S], [source[2] for source in S], color='blue')
        ax.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], 'r--')

        # ax.legend(['Microphones', 'True source', 'Estimated source'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    else:
        del fig
        return

    plt.show()

if __name__ == '__main__':
    N = 2
    n = 4
    dt = 1e-2
    alpha = 1e-1    # Acceleration noise
    sigma = 1e-2    # Measurement noise

    M, source, path = simulate(N, n, dt, alpha, sigma, num=449)
    visualize(M, source, path)

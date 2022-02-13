import numpy as np
import math
import matplotlib.pyplot as plt
from stationary import nu, F, localize


def setup(N, n, alpha, beta):
    M = np.random.rand(n, N) - 0.5*np.ones((n, N))
    source = np.random.normal(0, 0.125, N)
    mag = np.linalg.norm(M - source, axis=1)
    e = np.array([np.random.normal(0, a*10e-5) for a in mag])
    T = mag/nu + e
    s0 = source + np.random.normal(0, 0.1, N)
    v0 = np.random.normal(0, beta, N)
    x0 = np.block([s0, v0])
    
    J = JF(x0, M)
    P0 = alpha**2*np.dot(J.T, J)

    return M, source, T, x0, P0


def JF(x, M):
    n, N = M.shape
    s = x[:N]
    ret = []
    
    for i in range(n):
        for j in range(i + 1, n):
            grad = []
            m_i = M[i,:]
            m_j = M[j,:]
            for k in range(2*N):
                if k < N:    
                    df_ijk = -(m_j[k] - s[k])/np.linalg.norm(m_j - s) + (m_i[k] - s[k])/np.linalg.norm(m_i - s)
                    grad.append(df_ijk)
                else:
                    grad.append(0)

            ret.append(grad)

    return np.array(ret)

def step(xold, Pold, M, T, dt, alpha, beta):
    n, N = M.shape
    A = np.block([[np.eye(N),     dt*np.eye(N)],
                  [np.zeros((N, N)), np.eye(N)]])

    B = np.block([[dt**2/2*np.eye(N)],
                  [dt*np.eye(N)     ]])

    C = np.block([np.eye(N), np.zeros((N, N))])

    Q = alpha**2*np.block([[dt**4/4*np.eye(N), dt**3/2*np.eye(N)],
                           [dt**3/2*np.eye(N), dt**2*np.eye(N)  ]])

    R = beta**2

    a = np.random.normal(0, alpha, N)
    v = np.random.normal(0, beta, N)

    xmid = np.dot(A, xold) + np.dot(B, a)
    H = JF(xmid, M)
    z = np.dot(C, xmid) # + v ???
    y = z - localize(M, T)

    Pmid = np.dot(A, np.dot(Pold, A.T)) + Q
    S = np.dot(H, np.dot(Pmid, H.T)) + R*np.eye(2*N)
    K = np.dot(Pmid, np.dot(H.T, np.linalg.inv(S)))
    # xnew = xold + np.dot(K, y)
    xnew = xold + np.dot(K, np.block([y, v]))
    Pnew = np.dot(np.eye(2*N) - np.dot(K, H), Pmid)

    return xnew, Pnew

def simulate(N, n, dt, alpha, beta, num=10):
    M, source, T0, x0, P0 = setup(N, n, alpha, beta)
    path = np.ndarray((num + 1, N))
    path[0,:] = x0[:N]
    T = T0
    x = x0
    P = P0
    for i in range(1, num + 1):
        T = T + np.random.normal(0, 1e-5, n)
        x, P = step(x, P, M, T, dt, alpha, beta)
        path[i,:] = x[:N]

    return M, source, path


def visualize(M, source, path):
    _, N = M.shape

    fig = plt.figure()

    if N == 2:
        plt.scatter(M[:,0], M[:,1], color='g', marker='x')
        plt.scatter([s[0] for s in path], [s[1] for s in path], [s[2] for s in path], color='orange')
        plt.xlabel('x')
        plt.ylabel('y')

    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(M[:,0], M[:,1], M[:,2], color='g', marker='x')
        # ax.scatter(source[0], source[1], source[2], color='b', linewidths=5)
        ax.scatter([s[0] for s in path], [s[1] for s in path], [s[2] for s in path], color='orange')

        # ax.legend(['Microphones', 'True source', 'Estimated source'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    else:
        del fig
        return

    plt.show()

if __name__ == '__main__':
    N = 3
    n = 4
    dt = 1e-3
    alpha = 1e-5
    beta = 1e-5

    M, source, path = simulate(N, n, dt, alpha, beta, num=100)
    visualize(M, source, path)
    # plt.plot()

    # M, source, T0, x0, P0 = setup(N, n, alpha, beta)


    # test = step(x0, P0, M, T0, dt, alpha, beta)
    # print(test)
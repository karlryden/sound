import numpy as np
import matplotlib.pyplot as plt
import math
from linear import nu, measure

def setup(n, m, alpha, sigma):
    M = np.random.rand(m, n) - 0.5*np.ones((m, n))
    source = np.random.normal(0, 0.125, n)
    mag = np.linalg.norm(M - source, axis=1)
    x0 = np.zeros(2*n + m)
    P0 = np.block([[alpha**2*np.eye(2*n), np.zeros((2*n, m))],
                   [np.zeros((m, 2*n)),  sigma**2*np.eye(m)]])

    return M, source, x0, P0

# def Jacobi(x, M):
#     n, N = M.shape
#     s = x[:N]
#     ret = []
    
#     for i in range(n):
#         for j in range(i + 1, n):
#             grad = []
#             m_i = M[i,:]
#             m_j = M[j,:]
#             for k in range(2*N):
#                 if k < N:    
#                     df_ijk = -(m_j[k] - s[k])/np.linalg.norm(m_j - s) + (m_i[k] - s[k])/np.linalg.norm(m_i - s)
#                     grad.append(df_ijk)
#                 else:
#                     grad.append(0)

#             ret.append(grad)

#     return np.array(ret)


def F(x, M):
    m, n = M.shape
    A = np.block([[np.eye(n),     dt*np.eye(n)],
                  [np.zeros((n, n)), np.eye(n)]])

    C = np.block([np.eye(n), dt*np.eye(n)])

    diffs = []

    for i in range(m):
        m_i = M[i,:]
        grad = []
        for j in range(n):
            grad.append(
                -1/nu*(m_i[j] - (x[j] + dt*x[j + n]))/np.linalg.norm(m_i - np.dot(C, x[:2*n]))
            )

        for j in range(n):
            grad.append(
                -dt/nu*(m_i[j] - (x[j] + dt*x[j + n]))/np.linalg.norm(m_i - np.dot(C, x[:2*n]))
            )

        for _ in range(m):
            grad.append(0)

        diffs.append(grad)

    return np.block([[A, np.zeros((2*n, m))],
                     [np.array(diffs)      ]])

def step(xold, Pold, M, T, dt, alpha, sigma):
    m, n = M.shape

    Q = alpha**2*np.block([[dt**4/4*np.eye(n), dt**3/2*np.eye(n)],
                           [dt**3/2*np.eye(n), dt**2*np.eye(n)  ]])

    R = sigma**2*np.eye(math.comb(m, 2))

    xpred = np.dot(A, xold)

    J = F(xpred, M)
    # y = localize(M, T)
    # e = y - np.dot(C, xpred)

    Ppred = np.dot(A, np.dot(Pold, A.T)) + Q
    S = np.dot(J, np.dot(Ppred, J.T)) + R
    K = np.dot(Ppred, np.dot(J.T, np.linalg.inv(S)))

    xnew = xold + np.dot(K, e)
    Pnew = np.dot(np.eye(2*n) - np.dot(K, J), Ppred)

    return xnew, Pnew



def simulate(N, n, dt, alpha, sigma, num=10):
    M, s, x, P = setup(N, n, sigma)
    path = np.ndarray((num, N))
    S = np.ndarray((num, N))

    v = np.zeros(N)

    for i in range(num):
        S[i,:] = s

        T = measure(M, s)# + np.random.normal(0, sigma, n)
        x, P = step(x, P, M, T, dt, alpha, sigma)

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
    n = 3
    m = 5
    dt = 1e-2
    alpha = 1e-1    # Acceleration noise
    sigma = 1e-2    # Measurement noise

    M, source, x0, P0 = setup(n, m, alpha, sigma)
    print(F(x0, M))

    # M, source, path = simulate(N, n, dt, alpha, sigma, num=4999)
    # visualize(M, source, path)

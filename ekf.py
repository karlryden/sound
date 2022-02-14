import numpy as np
from linear import nu, F, measure, setup

def Jacobi(x, M):
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


def step():

    H = Jacobi(xmid, M)
    y = z - np.dot(H, xmid)
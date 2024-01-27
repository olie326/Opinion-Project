import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

#normalized poisson distribution


# #random edges
def randEdge(n, temp):
    edge_dict = { i: [] for i in range(n) }

    for i in range(temp):
        targets = [j for j in random.sample(range(n), random.randint(0, n)) if j != i]
        edge_dict[i] += [j for j in targets]
        for j in targets:
            if [i] not in edge_dict[j]:
                edge_dict[j] += [i]

    edge_dict = {k: set(v) for k, v in edge_dict.items()}

    return edge_dict


# #weights
# W = np.random.uniform(0.5, 1.5, size=(n,n))

# #bias weights
# B = np.random.uniform(0.5, 1.5, size=(n))

# # print(edge_dict)

# def x(t):
#     return 

def neighbor(i, edge_dict, X0, eps):
    return [j for j in edge_dict.get(i) if np.absolute(X0[i] - X0[j]) < eps]


# def init_update(i):
#     wii = W[i, i]
#     bi = B[i]
#     si0 = np.sum([W[i, j] * agents[j] for j in neighbor(i, eps=2)])
#     di0 = np.sum( [W[i, j] for j in neighbor(i, eps=2)] )
#     xi0 = agents[i]
#     xi = (  wii * xi0 + np.power(xi0, bi) * si0  )  /  (  wii +  np.power(xi0, bi) * si0 + np.power((1 - xi0), bi) * (di0 - si0)  )
    
#     return xi

# Xr = [init_update(i) for i in range(len(agents))]

# def opinion_update(k, X):
#     X_n = np.zeros((k, n))
    

#     for k in range(k):
#         X_k = np.array([])

#         for i in range(n):
#             neighbor = [j for j in edge_dict.get(i) if np.absolute(X[i] - X[j]) < 2]
#             si_k = np.sum([W[i, j] * X[j] for j in neighbor])
#             di_k = np.sum( [W[i, j] for j in neighbor])
#             xi = (  W[i,i] * X[i] + np.power(X[i], B[i]) * si_k  )  /  (  W[i,i] +  np.power(X[i], B[i]) * si_k + np.power((1 - X[i]), B[i]) * (di_k - si_k)  )

#             X_k = np.append(X_k, xi).T
        
#         X_n[k] = X_k
#         X = X_k
#     return X_n
# y = opinion_update(20, Xr)

def neighbor(i, edge_dict, X0, eps):
    return [j for j in edge_dict.get(i) if np.absolute(X0[i] - X0[j]) < eps]


def opnion_model(n, X0, eps, k):

    edge_dict = randEdge(n, 5)

    #weights
    W = np.random.uniform(0.5, 1.5, size=(n,n))

    #bias weights
    B = np.random.uniform(0.5, 1.5, size=(n))

    #Initial Update
    Xr = np.zeros(n)
    for i in range(len(X0)):
        wii = W[i, i]
        bi = B[i]
        si0 = np.sum([W[i, j] * X0[j] for j in neighbor(i, edge_dict, X0, eps) ])
        di0 = np.sum( [W[i, j] for j in  neighbor(i, edge_dict, X0, eps)] )
        xi0 = X0[i]
        xi = (  wii * xi0 + np.power(xi0, bi) * si0  )  /  (  wii +  np.power(xi0, bi) * si0 + np.power((1 - xi0), bi) * (di0 - si0)  )
        Xr[i] = xi

    X_n = np.zeros((k+1, n))
    X_n[0] = Xr   

    #Update

    for k in range(k):
        X_k = np.array([])

        for i in range(n):
            neighbors = [j for j in edge_dict.get(i) if np.absolute(Xr[i] - Xr[j]) < 2]
            si_k = np.sum([W[i, j] * Xr[j] for j in neighbors])
            di_k = np.sum( [W[i, j] for j in neighbors])
            a = W[i,i]
            b = Xr[i]
            c = B[i]
            d = (  a * b + np.power(b, c) * si_k  )  /  (  a +  np.power(b, c) * si_k + np.power((1 - b), c) * (di_k - si_k)  )
            xi_n = (  W[i,i] * Xr[i] + np.power(Xr[i], B[i]) * si_k  )  /  (  W[i,i] +  np.power(Xr[i], B[i]) * si_k + np.power((1 - Xr[i]), B[i]) * (di_k - si_k)  )

            X_k = np.append(X_k, xi_n).T
        
        X_n[k+1] = X_k
        Xr = X_k

    return X_n


poisson = stats.poisson.rvs(mu=100, size=50)
agents = (poisson - min(poisson)) / (max(poisson) - min(poisson))




y = opnion_model(50, agents, 2, 20)
ex = np.linspace(0, 21, 21)

for i in range(50):
    # plt.scatter(np.linspace(i, i, 500), y[i], marker='.')
    plt.plot(ex, y[:,i])



plt.show()




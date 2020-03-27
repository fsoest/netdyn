import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# %%
alpha = 0.09
beta = 0.01
epsilon = 0.01

j_0 = epsilon
s_0 = 1 - epsilon
r_0 = 0

y_0 = np.asarray([s_0, j_0, r_0])

def f(t,y):
    s, j, r = y
    return np.asarray([-alpha*s*j, alpha*s*j-beta*j, beta*j])

solution = solve_ivp(f, [0, 1000], y_0, method='LSODA')
plt.plot(solution['t'], solution['y'][0], label='susceptible')
plt.plot(solution['t'], solution['y'][1], label='infected')
plt.plot(solution['t'], solution['y'][2], label='recovered')
plt.xlabel('time')
plt.ylabel('proportion population')
plt.legend()
plt.title(r'$\alpha$ = {0}, $\beta$ = {1}, $\epsilon$ = {2}'.format(alpha, beta, epsilon))
plt.savefig('local_epidemic.png', dpi=300)
# %%
# Global pandemic
import pandas as pd
countries = pd.read_csv('countriesToCountries.csv')
# %%
import networkx as nx
g = nx.DiGraph()
g.add_nodes_from(countries['country departure'].unique())
for i in range(len(countries)):
    g.add_edge(countries['country departure'].iloc[i], countries['country arrival'].iloc[i], weight=countries['number of routes'].iloc[i])
#nx.draw(g)
# %%
adj_matrix = nx.to_numpy_matrix(g)
# Set self links to zero
for i in range(np.shape(adj_matrix)[0]):
    adj_matrix[i,i] = 0
# %%
adj_matrix = adj_matrix.astype(np.float64)
for i in range(np.shape(adj_matrix)[0]):
    adj_matrix[i] = adj_matrix[i] / adj_matrix[i].sum()

# %%
weighted = 1-np.log(adj_matrix)
weighted = np.where(weighted==np.inf, 0, weighted)
ng = nx.from_numpy_matrix(weighted, create_using=nx.DiGraph, parallel_edges=False)

dijkstra = dict(nx.all_pairs_dijkstra_path_length(ng))
dijkstra[223][4]

# %%
alpha = 0.02
beta = 0.002
T = 3000
epsilon = 0.003
gamma = 0.1
def ivp_function(t, y, alpha=alpha, beta=beta, gamma=gamma, epsilon=epsilon):
    # y = [S,J]
    S = y[:227]
    J = y[227:]
    dSdt = -1 * alpha * J * S * np.heaviside(J/epsilon, 0.5) + gamma * (adj_matrix - np.identity(np.shape(adj_matrix)[0])) @ S.T
    dJdt = alpha * J * S * np.heaviside(J/epsilon, 0.5) + (gamma*adj_matrix - (gamma+beta)*np.identity(np.shape(adj_matrix)[0])) @ J.T

    #dJdt = alpha*S*J*np.heaviside(J/epsilon, 0.5) + np.array(np.matmul(gamma*P - (beta+gamma)*np.identity(np.shape(P)[0]), J))[0]
    #dSdt = -alpha*S*J*np.heaviside(J/epsilon, 0.5) + gamma*np.array(np.matmul( P - np.identity(np.shape(P)[0]) , S))[0]
    return np.append(np.asarray(dSdt), np.asarray(dJdt))
# %%
def ivp_initial(country, epsilon):
    y_0 = np.zeros(2*len(countries['country departure'].unique()))
    for j in range(len(countries['country departure'].unique())):
        y_0[j] = 1
    i = np.where(countries['country departure'].unique()==country)[0].item()
    y_0[i] = 1-epsilon
    y_0[i + len(countries['country departure'].unique())] = epsilon
    return y_0

# %%
alpha = 2.5*0.07
beta = 0.07
T = 3000
epsilon = 0.01
gamma = 2.8e-3
solution = solve_ivp(ivp_function, (0, 1000), ivp_initial('China', epsilon), method='LSODA')
# %%
#for i in range(0, 227, 7):
plt.plot(solution['t'], solution['y'][1], label='s')
plt.plot(solution['t'], solution['y'][228], label='j')

# %%
np.shape(ivp_initial('China', epsilon))
np.shape(ivp_function(3, ivp_initial('China', epsilon)))
np.shape(((gamma*adj_matrix - (gamma+beta)*np.identity(np.shape(adj_matrix)[0])) @ y_0[227:].T))
len(y_0[int(len(y_0)/2):])
len(y_0[:227])

a = np.asarray([1,2])
b = np.asarray([3,4])
np.append(a,b)

# %%

np.where(countries['country departure'].unique()=='China')[0].item()

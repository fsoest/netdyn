import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
#%%
#startwerte
epsilon = 0.01
alpha = 0.5
beta = 0.1

#anfangsbedingungen
j0 = epsilon
s0 = 1 - epsilon
r0 = 0
y0 = np.asarray([s0, j0, r0])


def func(t,y):
    """gibt gekoppelte dgl als array zurück"""

    s,j,r = y

    dsdt = - alpha * s * j
    djdt = alpha * s * j - beta * j
    drdt = beta * j
    return np.asarray([dsdt, djdt, drdt])

solution = solve_ivp(func, [0,100], y0,method='LSODA')


plt.plot(solution['t'], solution['y'][0], label='susceptible')
plt.plot(solution['t'], solution['y'][1], label='infected')
plt.plot(solution['t'], solution['y'][2], label='recovered')
plt.xlabel('time')
plt.ylabel('population')
plt.title(r'$\alpha$ = {0}, $\beta$ = {1}, $\epsilon$ = {2}'.format(alpha, beta, epsilon))
plt.legend()


"""global pandemic"""
#daten lesen
countries = pd.read_csv('countriesToCountries.csv')
#erstelle directed graph
g = nx.DiGraph()
#füge dem graphen knoten hinzu, jedes land aber nur einmal
g.add_nodes_from(countries['country departure'].unique())
#gebe den kanten gewichte, gewicht ist anzahl der verbindungen
for i in range(len(countries)):
    g.add_edge(countries['country departure'].iloc[i], countries['country arrival'].iloc[i], weight = countries["number of routes"].iloc[i])

#lasse die self-links raus; heißt diagonal einträge der adjancency matrix auf null setzen
A = nx.to_numpy_matrix(g)
for i in range(len(countries["country departure"].unique())):
    A[i,i] = 0

#erstelle flux-fraction Matrix
P_mn = np.zeros_like(A)
for country in range(len(A[:,0])):
    for destination in range(len(A[0])):
        P_mn[country, destination] = A[country, destination]/np.sum(A[country])

#effective distances

D = 1 - np.log(P_mn)

#neuen graphen erstellen; mit d als gewichtung

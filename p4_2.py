import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%%
"""Anfangsbedingungen"""
Inf_0 = 1         #Anfangsinfektionen
N = 100         # Gesamtpopulation
eps = Inf_0/N   # Anteil von anfangs Infizierten
j_0 = eps
s_0 = 1-eps
r_0 = 0
y_0 = [s_0, j_0, r_0]
alpha = 0.09     # Transmission Rate
beta = 0.01      # Recovery Rate
#%%
def func_to_solve(t, y):
    """y = [s,j,r]"""
    s, j, r = y
    y1 = -alpha*s*j
    y2 = alpha * s*j -beta*j
    y3 = beta*j
    return([y1,y2,y3])

solution = solve_ivp(func_to_solve, [0,1000], y_0, method='LSODA')

#plt.plot(solution['t'], solution['y'][0],label='susceptible')
#plt.plot(solution['t'], solution['y'][1],label='infected')
#plt.plot(solution['t'], solution['y'][2],label='recovered')
#plt.xlabel('time')
#plt.ylabel('population')
#plt.legend()
#plt.title(r'$\alpha$ = {0}, $\beta$ = {1}, $\epsilon$ = {2}'.format(alpha, beta, eps))
#plt.savefig('Disease_spreading3.png', dpi = 300)
#%%
"""Global Pandemic"""

"""Daten einlesen"""

import pandas as pd
countries = pd.read_csv('countriesToCountries.csv')
countries.iloc[1]
countries.iloc[1]['country departure']
countries.iloc[1]['country arrival']
countries.iloc[1]['number of routes']
list(countries.iloc[2])
#%%
"""Weighted Directed Graph"""
np.array(countries)
import networkx as nx
g = nx.DiGraph()
g.add_nodes_from(countries['country departure'].unique())
for i in range(len(countries)):
    g.add_edge(countries['country departure'].iloc[i], countries['country arrival'].iloc[i], weight=countries['number of routes'].iloc[i])
#nx.draw(g)
#%%
"""Adjacency Matrix"""
adj_matrix = nx.to_numpy_matrix(g)                             # adj Matrix erstellen
l = len(countries['country departure'].unique())
for i in range(l):   # Travel zwischen selben Staaten auf 0
    adj_matrix[i,i] = 0
P_mn = np.zeros((l,l))
np.sum(adj_matrix[0])
"""Flux Fractions"""
for i in range(l):
    P_mn[i] = adj_matrix[i]/np.sum(adj_matrix[i])
np.sum(P_mn[20])
#P_mn[i]
"""Effective Length Matrix"""
Eff = 1-np.log(P_mn)                                    # Effective length
Eff = np.where(Eff==np.inf, 0, Eff)
g2 = nx.from_numpy_matrix(Eff, create_using=nx.DiGraph(), parallel_edges='false')

dijkstra = dict(nx.all_pairs_dijkstra_path_length(g2, weight= 'weight'))
#%%
"""Disease spreading"""
gamma = 2.8*10**(-3)

def DGL_worldwide(t,y):
    """Differentialgleichung für Disease spreading"""
    """y = [s1,j1]
            [s2,j2]
            [s3,j3]"""
    j = y[:l]
    s = y[l:]
    lsg = np.zeros(2*l)
    #y1 = []
    y1 = (alpha*s*j*np.heaviside(j/eps, 0.5) - beta*j + j@P_mn -j)
    #(gamma*P_mn-(gamma+beta)*np.eye(l)) @ j
    y2 = (-alpha*s*j*np.heaviside(j/eps, 0.5) + gamma * (j@P_mn-s))

    #dj = alpha*s*j*np.heaviside(j/eps, 0.5) + np.array(np.matmul(gamma*P_mn - (beta+gamma)*np.identity(np.shape(P_mn)[0]), j))[0]
    #ds = -alpha*s*j*np.heaviside(j/eps, 0.5) + gamma*np.array(np.matmul( P_mn - np.identity(np.shape(P_mn)[0]) , s))[0]
    #lsg = np.zeros(2*l)
    lsg[:l] = y1
    lsg[l:] = y2
    return lsg

def DGL_initial(country, eps):
    """Anfangskonditionen"""
    y_0 = np.zeros(2*l)
    i = np.where(countries['country departure'].unique()==country)[0].item()
    y_0[l:] = 1
    y_0[i] = eps
    y_0[i+l] = 1-eps
    return y_0
#%%
a = DGL_initial('China', 0.01)
a
"""ausrechnen"""
alpha = 2.5*0.07
beta = 0.07
gamma = 2.8e-3
epsilon = 10**(-6)

lösung = solve_ivp(DGL_worldwide, (0, 1000), DGL_initial('China', 0.01), method='LSODA')
#len(countries['country departure'].unique())
lösung
#%%
DGL_worldwide(1, DGL_initial('China', 0.01))

#%%

index = 50
t = lösung['t']
j = lösung['y'][index]
s = lösung['y'][index+l]
print

plt.figure()
plt.plot(t[0:100],s[0:100],label="susceptibles")
plt.plot(t[0:100],j[0:100],label="infected")
plt.xlabel("Time")
plt.ylabel("Fraction")
plt.legend()

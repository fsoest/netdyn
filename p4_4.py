import numpy as np
import matplotlib.pyplot as plt
import networkx as nwx
import scipy.integrate as sci



"""
task 1:
data structure
desease = [s(t),j(t),r(t)]
"""
#alpha = 0.02
#beta = 0.002
#T = 3000
#epsilon = 0.003
#
#def spread(t, desease):
#    s,j,r = desease
#    ds = -alpha*s*j
#    dj = alpha*s*j - beta*j
#    dr = beta*j
#    return np.array([ds,dj,dr])
#
#def initialspread(epsilon):
#    j0  = epsilon
#    s0 = 1 - j0
#    r0 = 0.
#    return np.array([s0,j0,r0])

#solution = sci.solve_ivp(spread, (0,T), initialspread(epsilon), method= 'LSODA')
#t = solution['t']
#s = solution['y'][0]
#j = solution['y'][1]
#r = solution['y'][2]
#
#plt.figure()
#plt.plot(t,s,label="susceptibles")
#plt.plot(t,j,label="infected")
#plt.plot(t,r,label="recovered")
#plt.xlabel("Time")
#plt.ylabel("Fraction")
#plt.legend()


# Global pandemic
import pandas as pd
countries = pd.read_csv('countriesToCountries.csv')

import networkx as nx
g = nx.DiGraph()
g.add_nodes_from(countries['country departure'].unique())
for i in range(len(countries)):
    g.add_edge(countries['country departure'].iloc[i], countries['country arrival'].iloc[i], weight=countries['number of routes'].iloc[i])
#nx.draw(g, with_labels = True)

adj_matrix = nx.to_numpy_matrix(g)
# Set self links to zero
for i in range(np.shape(adj_matrix)[0]):
    adj_matrix[i,i] = 0

P = np.zeros_like(adj_matrix)

for country in range(np.shape(adj_matrix)[0]):
    P[country] = adj_matrix[country]/np.sum(adj_matrix[country])

weighted_distances = 1 - np.log(P)
weighted_distances = np.where(weighted_distances==np.inf, 0, weighted_distances)

airtravel_nw = nx.convert_matrix.from_numpy_matrix(weighted_distances, create_using = nx.DiGraph, parallel_edges=False)
#nx.draw(airtravel_nw)

sp = nx.all_pairs_dijkstra_path_length(airtravel_nw, cutoff=None, weight='weight')

def shortestpath(weighted_graph, source, target):
    sp = dict(nx.all_pairs_dijkstra_path_length(airtravel_nw, cutoff=None, weight='weight'))
    length = sp[source][target]
    return length

sp_matrix = dict(sp)


"""
global network
data strucuture:
[ [s1,j1], [s2,j2], ...]
"""

alpha = 0.02
beta = 0.002
T = 3000
epsilon = 0.003
gamma = 0.1
N = np.shape(adj_matrix)[0]     #number of countries
k = 1       #first  country

"""one country:"""

def spread(t, desease):
    s,j = desease
    ds = -alpha*s*j
    dj = alpha*s*j - beta*j
    return np.array([ds,dj])

def initialspread(epsilon):
    j0  = epsilon
    s0 = 1 - j0
    return np.array([s0,j0])

def initialbreakout(index,epsilon = epsilon):
    initial = np.zeros((N,2))
    initial[index] = initialspread(epsilon)
    return np.reshape(initial, 2*N)

def globalspread(t,spread):
    spread = np.reshape(spread, (N,2))
    s = spread[:,0]
    j = spread[:,1]

    dj = alpha*s*j*np.heaviside(j/epsilon, 0.5) + np.array(np.matmul(gamma*P - (beta+gamma)*np.identity(np.shape(P)[0]), j))[0]
    ds = -alpha*s*j*np.heaviside(j/epsilon, 0.5) + gamma*np.array(np.matmul( P - np.identity(np.shape(P)[0]) , s))[0]

    return np.array(np.matrix([ds,dj]).T.flatten())[0]


solution = sci.solve_ivp(globalspread, (0,T), initialbreakout(k), method = 'LSODA')

def printspread(index):
    t = solution['t']
    s = solution['y'][2*index]
    j = solution['y'][2*index+1]

    plt.figure()
    plt.plot(t,s,label="susceptibles")
    plt.plot(t,j,label="infected")
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.legend()

printspread(1)

import numpy as np
import random
import matplotlib.pyplot as plt
#%%

def initial (N, L, v_max):
    """Zufällige Anfangswerte erstellen
    x = [x1, x2, x3, ....]
    v = [v1, v2, v3, ....]
    """
    #x = np.zeros(L)
    #v = np.zeros(L)
    x_0 = np.array(random.sample(range(0,L-1), N))
    v_0 = np.array([random.randint(0,v_max) for i in range(N)])
    #x[x_0] = 1
    #v[x_0] = v_0
    return x_0, v_0

def road(N, L, x):
    """Straße erstellen
    road = [0,0,1,0,0,1,1,0,...]
    """
    road = np.zeros(L)
    road[x] = 1
    return(road)

def movement(x, v,L):
    """Bewegung um Geschwindigkeit v
    new_x = [x1+v1, x2+v2, x3+v3, ....]"""
    new_x = (x + v) % L
    return(new_x)

def abstand(x, L):
    """Gibt abstand vom Vordermann für jedes Auto zurück
    x = [x1, x2, x3, ....]
    abst = [d1, d2, d3, ....]"""
    y = np.array(list(enumerate(x)))            # 2dim Array mit Nummerierung
    sort = np.array(sorted(y, key = lambda y: y[1]))     # sortiert Ort X
    d_zw = sort[:,1]-np.roll(sort,2)[:,1]       # Zwischenwert für d, da noch um 1 verschoben
    d_zw[0] += L
    d = np.roll(d_zw,-1)                        # Abstand unverschoben
    sort[:,1] = d
    d_sort = np.array(sorted(sort, key = lambda sort: sort[0]))
    abst = d_sort[:,1]
    return(abst)

def new_v(x,v,L,p,v_max):
    #d = abst(x,L)
    v_new = v
    for i in range(len(x)):
        if v_new[i] < v_max:             # acceleration
            v_new[i] += 1
        if abstand(x,L)[i] <= v[i]:                 # deceleration
            v_new[i] = abstand(x,L)[i]-1
        if np.random.uniform(0, 1) < p and v_new[i] > 0: # randomization
            v_new[i] -= 1
    return(v_new)
def flow(v_gesamt, T):
    return(np.sum(v_gesamt))

# %%
v_max = 5       # Maximalgeschwindigkeit
L = 100       # Streckenlänge
N = 20          # Anzahl autos
T = 10        # gesamte Simulationszeit
p = 0.2         # Randomness
"""Plot"""
x,v = initial(N,L,v_max)
x_gesamt = []
v_gesamt = 0
for t in range(T):
    v = new_v(x,v,L,p,v_max)
    v_gesamt = v_gesamt + np.sum(v)
    x = movement(x, v,L)
    x_gesamt.append(x)


for i in range(N):
    plt.scatter(np.arange(T), np.array(x_gesamt)[:,i], marker='.')
plt.xlabel('time step')
plt.ylabel('space')
#%%

initial(1,L,v_max)

"""Calculation of Flow"""
points = 100
flow = []
density = []
p_plot = []
for i in range (6):
    p=i*0.2
    for i in range(points-1):
        N = int((i+1)* (L/points))
        x,v = initial(N,L,v_max)
        #x_gesamt = []
        v_gesamt = 0
        for t in range(T):
            v = new_v(x,v,L,p,v_max)
            v_gesamt = v_gesamt + np.sum(v)
            x = movement(x, v,L)
            #x_gesamt.append(x)
            flow.append(v_gesamt / (L*T))
            density.append(N/L)
        p_plot.append()

plt.plot(density, flow[0:99])
plt.plot(density, flow[100:199])
plt.plot(density, flow[200:299])
plt.plot(density, flow[300:399])
plt.plot(density, flow[400:499])
plt.plot(density, flow[500:599])








#%%
"""Test"""
x, v = initial(N,L,v_max)
x
v
x+v
x = movement(x,v,L)
x
road(N,L,x)
abstand(x,L)
new_v(x,v,L,p,v_max)
np.sum(v_gesamt[2])
# %%
import gif
@gif.frame
frames = []
for t in range(T):
    plt.xlim(0,100)
    frame = plt.scatter(x_gesamt[t], 1)
    frames.append(frame)

# %%
gif.save(frames, 'animation.gif', duration=100)

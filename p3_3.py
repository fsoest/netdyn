import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

v_max = 5    # Maximalgeschwindigkeit
L = 100     # Straßenlänge
T = 100     # Simulationszeit total
N = 20       # Anzahl Autos <<L !!!
p = 0.2      # Random decel


"""
x = [x1, x2, x3, ....]
v = [v1, v2, v3, ....]
road = [0,0,1,0,0,1,1,0,...]
"""

def initial_posdistr(N,L):
    x = []
    i=0
    while i <N:
        xnew = rd.randint(0,L)
        if xnew in x: continue
        else:
            x.append(xnew)
            i += 1
    return np.array(x)

def initial_veldistr(N,v_max):
    v = rd.randint(0,v_max+1, size=N)
    return np.array(v)

def road(x,L):
    road = np.zeros(L)
    for i in x:
        road[i] = 1

    return np.array(road)

def dist(x,L):
    y = np.array(list(enumerate(x)))
    X = np.array(sorted(y, key = lambda position: position[-1]))
    vorne = X[1:,1]
    hinten = X[:-1,1]
    dist = vorne - hinten
    dist_lastcar = X[0,1]+ L - X[-1,1]
    dist = np.append(dist, dist_lastcar)
    X = np.transpose(X)
    X[1]=dist
    X = np.transpose(X)
    Distances = np.array(sorted(X, key = lambda position: position[0]))[:,1]
    return Distances


def update(x, v, L, N, v_max, p):

    Dist = dist(x,L)

    #changing velocity
    for i in range(N):
        #random brake
        w = rd.rand()
        if w<p and v[i] > 0:
            v[i] -= 1
        #brake
        if v[i] >= Dist[i] and v[i] > 0:
            v[i] = Dist[i] - 1
        #acceleration
        elif v[i] < v_max:
            v[i] += 1
    #moving on (with velocities of last step)
    x = (x+v)%L
    return x, v

def letsgo(N,L,v_max,p,T):
    t = np.arange(1,T)


    x = initial_posdistr(N,L)
    v = initial_veldistr(N,v_max)
    motion = [[x,v]]

    for tau in t:
        x, v = update(x,v, L,N,v_max,p)
        motion.append([x,v])
    motion = np.array(motion)

    X = motion[:,0]
    V = motion[:,1]
    t = np.arange(0,T)

    return motion


def plottraffic(motion):
    X = motion[:,0]
    t = np.arange(0,len(X))
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("position")
    for car in range(N):
        plt.plot(t,X[:,car],'.')


motion = letsgo(N,L,v_max,p,T)
plottraffic(motion)


def flow(motion,L):
    V = motion[:,1]
    N = len(V[0])
    v_mean = np.array([np.mean(V[:,car]) for car in range(N)])
    flow = sum(v_mean)/L
    return flow

print(flow(motion,L))

def animateroad(motion):
    X = motion[:,0]
    t = np.arange(0,len(X))
    H = np.ones(N)

    plt.figure()
    plt.subplot(111)
    plt.xlim(0,L)

    for tau in t:
        plt.pause(0.001)
        plt.subplot(111).clear()
        plt.plot(X[tau],H,'.')
        plt.xlim(0,L)

def fundamentaldiag(N_max,L,p,v_max = v_max, T = T):
    Flow = []
    n = np.arange(1,N_max+1)
    for N in n:
        Flow.append(flow(letsgo(N,L,v_max,p,T),L))
    n = np.arange(1,N_max+1)
    Flow = np.array(Flow)
    #print diagram
#    plt.figure()
#    plt.plot(n, Flow)
#    plt.title("Fundamental Diagram")

    return n, Flow

plt.figure()
for p in np.linspace(0.1,1,11):
    n, f = fundamentaldiag(L,L,p)
    plt.plot(n,f,label="p = {}".format(p))

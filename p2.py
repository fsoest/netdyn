
import numpy as np
from scipy.integrate import solve_ivp
# %%
L_x = 20    # lane length
L_y = 5     # lane width
A = 0.2       # reasonable model parameter: A=0.2
B = 2       # reasonable model parameter: B=2
D = 1       # hardcore diameter
N = 20      # number of walkers
Nr = 2      # pedestrians walking to the right
Nl = 2      # pedestrians walking to the left
tf = 20     # Final IVP Time
ti = 0      # Initial IVP Time
fstep = 1000 # Fluctuation step


# %%
def gaussian(theta):
    """distribution truncated: abs(xi(t)) <= 3 sqrt(theta)"""
    return min(3*np.sqrt(theta), np.random.normal(0, np.sqrt(theta)))

class ped:
    # reasonable model parameters: tau=0.2, v0=1, m=1
    def __init__(self, drive_direction, theta, v_pref=1, tau=0.2, m=1):
        # Value for ivp
        self.x_0 = np.asarray([np.random.uniform(L_x/4,3*L_x/4), np.random.uniform(L_y/4,3*L_y/4)])
        # Value for ivp
        self.v_0 = np.asarray([np.random.uniform(-1,1),np.random.uniform(-1,1)])
        # Preferred walking speed
        self.v_pref = v_pref
        # relaxation time
        self.tau = tau
        self.m = m
        # drive direction vector
        self.e = np.asarray([drive_direction, 0])
        # Array in dem zufallswerte Stehen
        self.fluctuations = []
        # fluctuations as piece-wise constant function with stepwidth fstep
        for i in np.linspace(ti, tf, fstep):
            # radial definition
            phi = np.random.uniform(0, 2*np.pi)
            # normierung ..
            self.fluctuations.append(np.asarray([gaussian(theta)*np.sin(phi), gaussian(theta)*np.cos(phi)])/np.sqrt(1/fstep))

    def interaction_force(self, x_own, x_other):
        """repulsive interaction zwischen selbst und übergebenem pedestrian"""
        if x_own[0] == x_other[0] and x_own[1] == x_other[1]:
            return 0
        else:
            return A * B * (np.abs(x_own-x_other) - D)**(-B-1) * (x_own - x_other) / np.abs(x_own-x_other)

    def xi(self, t):
        """gibt bei geg. t die entsprechende Stufe der stufenweisen Funktion xi
        aus, Stufenbreite: fstep, Stufe jeweils nach unten gerundet"""
        return self.fluctuations[int(np.floor(t/fstep))]

    def boundary_force(self, x):
        """d distance to the closest boundary"""
        d = min(abs(x[1]), abs(L_y - x[1]))
        # Richtung der Kraft
        if x[1] < L_y/2:
            # when ped in lower half of sim.box, forced upwards
            # d must not < D/2 == hard wall potential
            return 1*A * B * (d - D/2)**(-B-1) * (np.asarray([0,d])) / np.abs(d)
        elif x[1] > L_y/2:
            # when ped in upper half of sim.box, forced downwards
            return -1 * A * B * (d - D/2)**(-B-1) * (np.asarray([0,d])) / np.abs(d)
        else:
            return 0

    def motion(self, t, all, v, x_own, x_other):
        """calculates acceleration as sum of
        friction term, stochastic term, hard-core interaction and boundary interaction"""
        # driving force
        d_f = self.m/self.tau*(self.v_0*self.e-v)
        # fluctuations
        fluc = self.xi(t)
        # aufsummierte repulsive interaction zwischen jedem pedestrian
        rep = 0
        for x in x_other:
            rep = rep + self.interaction_force(x_own, x)
        bnd = self.boundary_force(x_own)
        return((d_f + fluc + rep + bnd)/self.m)
        #return x_other
# %%


theta = 25
def make_pedestrians(Nl, Nr):
    """gibt jedem pedestrian eine ID in all
    zuerst alle nach rechts Gehenden, dann alle nach links"""
    all = []
    for i in range(Nr):
        all.append(ped(1, theta))
    for i in range(Nl):
        all.append(ped(-1, theta))
    return(all)


def function_for_ivp(t, y):
    """ Takes y = [x1,x2,v1,v2,x1,x2,v1,v2,...]
        Returns x- and y-direction velocities and accelerations
            dy/dt = [(v, dv/dt)] = f(t,y) = [v1,v2,a1,a2,v1,v2,a1,a2,...]"""
    to_return = []
    x = []
    for i in range(len(all)):
        # make x = [x1,x2, x1,x2, ...]
        x.append(np.asarray([y[4*i], y[4*i+1]]))
    for i in range(len(all)):
        if x[i][0] > L_x:
            x[i][0] = x[i][0] % L_x
        if x[i][0] <= 0:
            x[i][0] = L_x - (x[i][0] % L_x)
        to_return.append(y[4*i+2])          # append v1 for pedestrian i
        to_return.append(y[4*i+3])          # append v2 for pedestrian i
        motion = all[i].motion(t, all, np.asarray(y[4*i+2], y[4*i+3]), x[i], x)
        to_return.append(motion[0])         # append a1 for pedestrian i
        to_return.append(motion[1])         # append a2 for pedestrian i

    return to_return


def initial_values_for_ivp(all):
    ivps = []
    for ped in all:
        ivps.append(ped.x_0[0])
        ivps.append(ped.x_0[1])
        ivps.append(ped.v_0[0])
        ivps.append(ped.v_0[1])
    return ivps


def ivp (ti, tf, all):
    """langes Array mit Lösungen für x1,x2,v1,v2"""
    solve_ivp(function_for_ivp, [ti, tf], initial_values_for_ivp(all))

# %%
Nl = 2
Nr = 0
all = make_pedestrians(Nl, Nr)
solution = solve_ivp(function_for_ivp, [ti, tf], initial_values_for_ivp(all), method='LSODA')
print(min(solution['y'][1]))
print(max(solution['y'][1]))
len(solution['y'][0])
solution
# %%
solution['y'][2,0]
# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(tight_layout = True)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, L_x), ylim=(0, L_y))


solution.y[4*i][0]

n_1 = Nl
n_2 = Nr
solution.y[4*(i+Nl)][0]
# Plot a scatter that persists (isn't redrawn) and the initial line.
x = []
y = []
for i in range(Nl):
    x = x.append(solution.y[4*i,0])
    y = y.append(solution.y[4*i+1,0])
for i in range(Nr):
    x = x.append(solution.y[4*(i+Nl),0])
    y = y.append(solution.y[4*(i+Nl)+1,0])
#%%
#particles_2, = ax.plot(solution[1][n_1:(n_1+n_2),0], solution[1][n_1:(n_1+n_2),1], "r.", markersize = 10)

particles_1 = ax.plot(x[:Nl], y[:Nl], "b.", markersize = 10)
particles_2 = ax.plot(x[Nl:], y[Nl:], "b.", markersize = 10)
def update(i):
    particles_1.set_xdata(solution.y[:n_1,0,i])
    particles_1.set_ydata(solution.y[:n_1,2,i])
    particles_2.set_xdata(solution.y[n_1:,0,i])
    particles_2.set_ydata(solution.y[n_1:,2,i])
    return particles_1, particles_2, ax


#anim = FuncAnimation(fig, update, frames=np.arange(0, n_t), interval=1/fps*1000)

import numpy as np
from scipy.optimize import solve_ivp

# %%
class ped:
    # reasonable model parameters: tau=0.2, v0=1, m=1
    def __init__(self, drive_direction, theta, v_pref=1, tau=0.2, m=1):
        # Value for ivp
        self.x_0 = np.asarray([np.random.uniform(0,L_x), np.random.uniform(0,L_y)])
        self.x = self.x_0
        # Value for ivp
        self.v_0 = np.asarray([np.random.uniform(-1,1),np.random.uniform(-1,1)])
        # Preferred walking speed
        self.v_pref = v_pref
        # relaxation time
        self.tau = tau
        self.m = m
        self.e = np.asarray([drive_direction, 0])
        # Array in dem zufallswerte Stehen
        self.fluctuations = []
        # fluctuations as piece-wise constant function with stepwidth fstep
        for i in np.linspace(ti, tf, fstep):
            self.fluctuations.append(gaussian(theta))

def make_pedestrians(Nr, Nl):
    """gibt jedem pedestrian eine ID in all
    zuerst alle nach rechts Gehenden, dann alle nach links"""
    all = []
    for i in range(Nr):
        all.append(ped(1, theta))
    for i in range(Nl):
        all.append(ped(-1, theta))
    return(all)

 def f(t, y):

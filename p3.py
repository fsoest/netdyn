# Nagel-Schreckberg-Modell
import numpy as np
import matplotlib.pyplot as plt
# %%
class Car:
    """Car class for implementation of iterations"""
    def __init__(self, x_start):
        self.x = []
        self.x.append(x_start)
        self.v = 0
        self.distance_traveled = 0

    def distance_to_other_car(self, car):
        calc_dist = car.x[-1] - self.x[-1]
        if calc_dist > 0:
            return calc_dist
        elif calc_dist < 0:
            return calc_dist + L
        else:
            return 1000

    def update(self):
        self.x.append((self.x[-1]+max(self.v,0)) % L)
        self.distance_traveled += max(self.v, 0)

def car_logic(cars, p=0.2):
    cars.sort(key=lambda car:car.x[-1])
    for i in range(len(cars)):
        # Acceleration
        if cars[i].v < v_max:
            cars[i].v += 1
            #print('Acceleration: Speed {0}'.format(cars[i].v))

        # Deceleration
        if i < len(cars)-1:
            distance = cars[i].distance_to_other_car(cars[i+1])
            if cars[i].x[-1] + cars[i].v >= cars[i].x[-1] + distance:
                cars[i].v = distance -1
                #print('Decel: Speed {0}'.format(cars[i].v))
        else:
            distance = cars[i].distance_to_other_car(cars[0])
            if cars[i].x[-1] + cars[i].v >= cars[i].x[-1] + distance:
                cars[i].v = distance - 1
                #print('distance: {0}'.format(distance))
                #print('Decel else: Speed {0}'.format(cars[i].v))


        # Randomisation
        if np.random.uniform(0, 1) < p:
            cars[i].v -= 1
            #print('Random: Speed {0}'.format(cars[i].v))
    # Movement
    for i in range(len(cars)):
        cars[i].update()

# %%
v_max = 5    # Maximalgeschwindigkeit
L = 100     # Straßenlänge
T = 100     # Simulationszeit total
N = 20       # Anzahl Autos
p = 0.2      # Random decel

# %%
cars = []
sample = np.random.choice(L, N, replace=False)
for i in sample:
    cars.append(Car(i))

for t in range(T):
    car_logic(cars)

for car in cars:
    plt.scatter(range(T+1), car.x, marker='.')
plt.xlabel('time step')
plt.ylabel('space')
plt.savefig('test.png', dpi=300)
# %%
# Calculate flow
total_distance = 0
for car in cars:
    total_distance += car.distance_traveled

# %%
v_max = 5    # Maximalgeschwindigkeit
T = 100     # Simulationszeit total
p = 0
L = 1000
n_for_plot = []
dist_for_plot = []
dist_normed = []
# Create fundamental diagram
for n in np.arange(0, 1000, 10):
    cars = []
    sample = np.random.choice(L, n, replace=False)
    for i in sample:
        cars.append(Car(i))
    for t in range(T):
        car_logic(cars)
    total_distance = 0
    for car in cars:
        total_distance += car.distance_traveled
    n_for_plot.append(n)
    dist_for_plot.append(total_distance)
    dist_normed.append(total_distance/n)
#%%
plt.plot(np.asarray(n_for_plot)/L, np.asarray(dist_for_plot)/(L*T), label='L*T')
#plt.plot(np.asarray(n_for_plot)/L, dist_normed, label='Normiert')
#plt.plot(np.asarray(n_for_plot)/L, np.asarray(dist_for_plot), label='Schritte')
plt.legend()
plt.xlabel('density (N/L)')
plt.ylabel('flow (distance traveled/(L*T))')
#plt.savefig('fundamental_lab.png', dpi=300)

# %%
# p-Dependance
v_max = 5    # Maximalgeschwindigkeit
T = 100     # Simulationszeit total
L = 100
n_for_plot = []
dist_for_plot = []
dist_normed = []

# %%
import pandas as pd
p_for_plot = []
# %%
for p in np.linspace(0,1,11):
    # Create fundamental diagram
    dist_for_plot = []
    n_for_plot = []
    for n in np.arange(0, 100, 1):
        cars = []
        sample = np.random.choice(L, n, replace=False)
        for i in sample:
            cars.append(Car(i))
        for t in range(T):
            car_logic(cars, p)
        total_distance = 0
        for car in cars:
            total_distance += car.distance_traveled
        n_for_plot.append(n)
        dist_for_plot.append(total_distance)
    p_for_plot.append(dist_for_plot)
#%%
for i in range(11):
    plt.plot(np.asarray(n_for_plot)/L, np.asarray(p_for_plot[i])/(L*T), label='p={0}'.format(np.round(i*0.1,1)))

plt.legend()
plt.xlabel('density (N/L)')
plt.ylabel('flow (distance traveled/(L*T))')
#plt.savefig('p-dependance.png', dpi=300)
# %%


# %%
v_max = 5    # Maximalgeschwindigkeit
L = 1000     # Straßenlänge
T = 100     # Simulationszeit total
N = 50       # Anzahl Autos
p = 0.2      # Random decel

# %%
cars = []
sample = np.random.choice(L, N, replace=False)
for i in sample:
    cars.append(Car(i))

for t in range(T):
    car_logic(cars)

for car in cars:
    plt.scatter(range(T+1), car.x, marker='.')
plt.xlabel('time step')
plt.ylabel('space')
plt.savefig('test.png', dpi=300)

# %%
# Animation
import gif
@gif.frame
def plot(t, cars):
    for car in cars:
        plt.scatter(car.x[t], 1)
        plt.xlim(0,L)

# %%
frames = []
for t in range(T):
    frame = plot(t, cars)
    plt.title('N = {0}'.format(N))
    frames.append(frame)

# %%
gif.save(frames, 'animation3.gif', duration=100)

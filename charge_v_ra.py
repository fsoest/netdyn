import simpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%

class PosEnvironment(simpy.Environment):
    def __init__(self, x_gesamt=0):
        self.x_gesamt = x_gesamt
        super().__init__()

    """Klasse die gleichzeitig timeout durchführt und den Ort updated"""
    def time_and_place(self, car, duration, id, L=100):
        if id == 'go':
            car.position = (car.position + duration) % L
            car.charge -= duration
            self.x_gesamt += 1
        if id == 'charge':
            while car.charge < car.max_charge:
                car.charge += 1
        return self.timeout(duration)


class Car():
    """Car class to implement model functionality
    x: List, keeps track of car position over time
    x_0: Integer, starting position of car
    """
    def __init__(self, env, x_0, charge, patience, range_anxiety, max_charge, CS, name='Bernd'):
        self.env = env
        self.position = x_0
        self.charge = charge
        self.patience = patience
        self.range_anxiety = range_anxiety
        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())
        self.max_charge = max_charge
        self.name = name
        self.CS = CS

    def run(self):

        while True:
            if self.charge > self.range_anxiety:
                # move to next station
                # print(' {0} leaving {1} at {2}min'.format(self.name, self.position, self.env.now))
                yield self.env.time_and_place(self, 1, 'go')


            elif self.CS[self.position].queue_length() > (self.patience-1) and self.charge > 0:
                # move to next station
                # print('{0} moves to next CS, because of queue at {1}'.format(self.name, self.position))
                yield self.env.time_and_place(self, 1, 'go')

            else:
                # print('{0} arriving {1} at {2}min'.format(self.name, self.position, self.env.now))
                with self.CS[self.position].resource.request() as req:
                    yield req

                    # Charge the battery
                    # print('%s starting to charge at %smin' % (self.name, self.env.now))
                    yield self.env.time_and_place(self, self.max_charge-self.charge, 'charge')

                    # print('{0} leaving bcs {1} at {2}min'.format(self.name, self.position, self.env.now))


class charging_station():
    def __init__(self, x, env, capacity):
        self.ort = x
        self.resource = simpy.Resource(env, capacity=capacity)

    def queue_length(self):
        return len(self.resource.queue) - (self.resource.capacity-len(self.resource.users))


def collect_x_data(env, x_data, q_data, T, cars, CS):
    """data = [[x0_1,x0_2,x0_3,x0_4...., x0_n],
              [x1_1,x1_2,x1_3x1_4......,x1_n]
              ...
              [xT_1, xT_2, xT_3, xT_4,....,xT_n]]"""
    for i in range(T):
        zeitpunkt_x = []
        zeitpunkt_q = []
        for car in cars:
            zeitpunkt_x.append(car.position)
        for station in CS:
            zeitpunkt_q.append(len(station.resource.queue))
        x_data.append(zeitpunkt_x)
        q_data.append(zeitpunkt_q)
        yield env.timeout(1)

def parameters(N, charging_speed, max_charge, patience, range_anxiety, plot=False, plot2=False, L=100):
    env = PosEnvironment()
    CS = []
    for i in range(L):
        CS.append(charging_station(i, env, 1))
    cars = []
    name = range(N)
    sample = np.random.choice(L, N, replace=True)
    #charge = max_charge
    i = 0
    for x0 in sample:
        charge = np.random.randint(0, max_charge)
        cars.append(Car(env, x0, charge, patience, range_anxiety, max_charge, CS, i))
        i += 1
    data = []
    q_data = []
    coll = env.process(collect_x_data(env, data, q_data, T, cars, CS))

    env.run(coll)
    if plot==True:
        #plt.figure(dpi=420)
        for i in range(len(data[0])):
            plt.scatter(range(len(data)), [data[j][i] for j in range(len(data))], marker='.', c='black', alpha=0.002)
            #plt.xlim(3000,4000)
            plt.xlabel('Zeit')
            plt.ylabel('Ort')
            plt.title('N={0}, Charging speed = {1}, Max charge = {2}, Patience = {3}, Range anxiety = {4}'.format(N, charging_speed, max_charge, patience, range_anxiety))
        plt.legend(title='Flow={0}'.format(env.x_gesamt/(L*T)))
        plt.plot(np.linspace(3500,3700, 1000), [i/2 for i in np.linspace(0,200,1000)], c='red')
        plt.plot(np.linspace(3000,3200, 1000), [i/2 for i in np.linspace(0,200,1000)], c='red')

    if plot2==True:
        for i in range(len(q_data[0])):
            plt.scatter(range(len(q_data)), [q_data[j][i] for j in range(len(data))], marker='.', c='black', alpha=0.01)
            plt.xlabel('Zeit')
            plt.ylabel('Anzahl in Schlange')

    return [N/L, env.x_gesamt/(L*T), patience, range_anxiety]

# %%
T = 5000
# seed = 43
N = 200
charging_speed = 1
r_a = 9
max_charge = 10
patience = 1
# %%
# Max_charge 10, r_a = 4
np.random.seed(seed)
parameters(N, charging_speed, max_charge, patience, r_a, plot=True)
# %%
# Max charge 6, r_a=0
np.random.seed(seed)
parameters(N, charging_speed, max_charge-r_a, patience, 0, plot=True)

# %%
data_without_ra = []
for i in range(10):
    for n in np.linspace(0, 350, 36):
        np.random.seed(i)
        data_without_ra.append(parameters(int(n), charging_speed, 10, patience, 0))
        print(i, n)
pd.DataFrame(data_without_ra, columns=['N/L', 'Flow', 'Pat', 'r_a']).to_csv('data_without_ra_10.csv')

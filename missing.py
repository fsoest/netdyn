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


            # elif self.CS[self.position].queue_length() > (self.patience-1) and self.charge > self.closest_station(self.CS):
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

    def closest_station(self, CS):
        for station in CS:
            if station.ort >= self.position:
                return station.ort-self.position
        return CS[0]-self.position+L


def congestion_speed(q_data):
    T, N = np.shape(np.matrix(q_data))
    q_data= np.matrix(q_data)
    c_speed=[]
    j=0
    k=3000
    while k<T and j<N:
        if np.array(q_data[k,j])>1:
            c_speed.append([k,j])
            j += 1
        else:
            k += 1
    return c_speed
    #return q_data[k,j]

def speed(c_speed):
    #x = c_speed[:,0]
    #y = c_speed[:,1]
    if len(c_speed) > 1:
        l = len(c_speed)/4
        a, b = np.polyfit(np.array(c_speed.T[0][0]).flatten()[int(l):], np.array(c_speed.T[1][0]).flatten()[int(l):], 1)
        return a, b
    else:
        return 0, 0

c = np.matrix([[5, 3], [5,7]])
x = np.array(c.T[0][0]).flatten()

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
              [xT_1, xT_2, xT_3, xT_4,....,xT_n]]

    q_data =  [[q0_1, q0_2, q0_3..., q0_n],
                [q1_1,...............q1_n],
                ...
                [qT_1,qT_2,qT_3,........,qT_4]]     """
    for i in range(T):
        zeitpunkt_x = []
        zeitpunkt_q = []
        for car in cars:
            zeitpunkt_x.append(car.position)
        for station in CS:
            zeitpunkt_q.append(len(station.resource.queue)+len(station.resource.users))
        x_data.append(zeitpunkt_x)
        q_data.append(zeitpunkt_q)
        yield env.timeout(1)
# %%
def parameters(N, charging_speed, max_charge, patience, range_anxiety, plot=False, plot2=False, plot3=False, L=100):
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
    stau = np.matrix(congestion_speed(q_data))
    a, b = speed(stau)
    print('Speed is {0}; flow is {1}'.format(a, env.x_gesamt/(L*T)))
    #speed= (stau[-1,1]-stau[0,1])/(stau[-1,0]-stau[0,0])
    if plot==True:
        for i in range(len(data[0])):
            plt.scatter(range(len(data)), [data[j][i] for j in range(len(data))], marker='.', c='black', alpha=0.01*200/N)

            plt.xlim(3000,4000)
            plt.xlabel('Zeit')
            plt.ylabel('Ort')
            plt.title('N={0}, Charging speed = {1}, Max charge = {2}, Patience = {3}, Range anxiety = {4}'.format(N, charging_speed, max_charge, patience, range_anxiety))
        plt.plot(stau[:,0], stau[:,1], c='red')
        plt.plot(np.linspace(int(- b /a) ,int ((100-b)/a), 1000), [a*i+b for i in np.linspace(int(- b /a) ,int ((100-b)/a), 1000)], c = 'green')
        plt.plot(np.linspace(3500,3700, 1000), [i/2 for i in np.linspace(0,200,1000)], c='red')
        plt.plot(np.linspace(3000,3200, 1000), [i/2 for i in np.linspace(0,200,1000)], c='red')

    if plot2==True:
        for i in range(len(q_data[0])):
            plt.scatter(range(len(q_data)), [q_data[j][i] for j in range(len(data))], marker='.', c='black', alpha=0.01)
            plt.xlabel('Zeit')
            plt.ylabel('Anzahl in Schlange')

    if plot3==True:
        np.matrix(q_data)
        for i in np.shape(q_data[0]):
            q_data[0]=np.sum(q_data[i])
        axs[2].plot(range(len(q_data)),q_data[:,0])
    return [N, L, env.x_gesamt/(L*T), patience, range_anxiety, a]
# %%



T = 5000
#seed = 1
#np.random.seed(seed)
q = parameters(N=210, charging_speed=1, max_charge=10, patience=1, range_anxiety=3, plot=True, plot2=False, plot3=False)
# %%

data_with_speed = []
for i in range(10):
    for n in np.linspace(10, 350, 35):
        data_with_speed.append(parameters(int(n), 1, 10, 1, 3))
        print(i, n)
pd.DataFrame(data_with_speed, columns=['N', 'L', 'Flow', 'Pat', 'r_a', 'speed']).to_csv('data_with_speed.csv')

# %%
#np.random.seed(seed)
#parameters(N=210, charging_speed=1, max_charge=10, patience=1, range_anxiety=3, plot=True)

#%%
np.random.seed(seed)
parameters(N=210, charging_speed=1, max_charge=10, patience=1, range_anxiety=3, plot=False, plot2=True, plot3=False)

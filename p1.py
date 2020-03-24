import numpy as np

def choices(a, b, c):
    return np.asarray([a, b, c])
def phi(a, b, c):
    return np.asarray([a+c, b, b+c, a, c])

path_matrix = np.matrix([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 1, 0, 1]])
cost_per_path = np.diag([10, 1, 10, 1, 1])
cost_add_term = np.asarray([0, 50, 0, 50, 10])

def cost(a, b, c):
    e = cost_per_path @ phi(a, b, c) + cost_add_term
    f = path_matrix @ e
    return np.dot(f,choices(a, b, c))
# %%
# User equilibrium
def phi(a, b, c):
    return np.asarray([a+c, b, b+c, a, c])

def linear(x, args):
    a, c = args
    return a*x + c

cost_funcs = ((10,0), (1, 50), (10,0), (1,50), (1, 10))

def potential(a,b,c):
    phi_tot = []
    for i in range(len(phi(a,b,c))):
        running_sum = 0
        for j in range(phi(a,b,c)[i]+1):
            running_sum += linear(j, cost_funcs[i])
        phi_tot.append(running_sum)
    return np.sum(phi_tot)
# %%
N = 5
best_vals = [N, N, N]
for a in range(N+1):
    for b in range(N-a+1):
        if potential(a, b, N-a-b) < potential(best_vals[0], best_vals[1], best_vals[2]):
            best_vals = [a, b, N-a-b]
best_vals
# %%
# Social optimum
def choices(a, b, c):
    return np.asarray([a, b, c])
def phi(a, b, c):
    return np.asarray([a+c, b, b+c, a, c])

path_matrix = np.matrix([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 1, 0, 1]])
cost_per_path = np.diag([10, 1, 10, 1, 1])
cost_add_term = np.asarray([0, 50, 0, 50, 10])


def func_to_min(params):
    a, b, c = params
    return cost(a, b, c)

N = 5
best_vals = [N, N, N]
for a in range(N+1):
    for b in range(N-a+1):
        if cost(a, b, N-a-b) < func_to_min(best_vals):
            best_vals = [a, b, N-a-b]
best_vals

cost(1,2,2)/cost(2,3,0)

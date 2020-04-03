from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
# %%
d1 = pd.read_csv('total1.csv')
d2 = pd.read_csv('total2.csv')
d1.drop('Unnamed: 0', axis=1, inplace=True)
d2.drop('Unnamed: 0', axis=1, inplace=True)

d3 = d1.append(d2)
# %%
total_y = d3['Y']
X_train, X_test, Y_train, Y_test = train_test_split(d3.drop('Y', axis=1), total_y, test_size=0.24)
# %%
net = MLPRegressor((200, 50, 10), )
net.fit(X_train, Y_train)
# %%
net.score(X_test, Y_test)

pd.DataFrame(d3['Y']).plot()
len(d3['Y'])
import matplotlib.pyplot as plt
plt.scatter(range(len(d3['Y'])), d3['Y'])
cats = []
for row in d3:
    

from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.cluster import KMeans
# %%
d1 = pd.read_csv('total1.csv')
d2 = pd.read_csv('total2.csv')
d1.drop('Unnamed: 0', axis=1, inplace=True)
d2.drop('Unnamed: 0', axis=1, inplace=True)

d3 = d1.append(d2)
# %%
total_y = d3['Y']
cluster = KMeans(4)
cluster.fit(np.array(total_y).reshape(-1,1))
# %%
net = MLPRegressor((200, 50, 10), )
net.fit(X_train, Y_train)
# %%
net.score(X_test, Y_test)


import matplotlib.pyplot as plt
plt.scatter(range(len(d3['Y'])), d3['Y'], c=cluster.predict(np.array(total_y).reshape(-1,1)))
plt.scatter(range(len(d3['Y'])), d3['Y'])

# %%
d3['cat'] =  cluster.predict(np.array(total_y).reshape(-1,1))
X_train, X_test, Y_train, Y_test = train_test_split(d3.drop(columns=['Y']), categories, test_size=0.23)
# %%
d3.iloc[4]
# %%
svc = SVC()
svc.fit(X_train, Y_train)

svc.score(X_test, Y_test)
svc.predict(X_test)
# %%
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier


d3.to_csv('total_with_cat.csv')

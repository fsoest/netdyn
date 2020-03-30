import pandas as pd
import matplotlib.pyplot as plt

# %%

data = pd.read_csv('data.csv')
data

# %%
plt.scatter(data['N']/100, data['Max'])
plt.xlabel('Density')
plt.ylabel('Flow')
# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%

data = pd.read_csv('data.csv')
data
# %%
plt.scatter(data['N']/100, data['Max'], marker='.')
plt.xlabel('Density')
plt.ylabel('Flow')
plt.title('Fundamental diagram, Patience=1, Range Anxiety=3, L=100')
# plt.xlim(1.7,2.5)
# plt.ylim(0.8, 1.01)
# plt.savefig('images/fundamental_diagram_zoom.png', dpi=420)
# %%
r_a = pd.read_csv('range_anxiety.csv')
r_a['Density'] = r_a['N']/100
r_a['Flow'] = r_a['Max']
ax = sns.catplot(x='Density', y='Flow', hue='R_A', data=r_a)
ax.fig.suptitle('Flow vs Density, varying Range anxiety, L=100, Patience=1')
# %%
pat = pd.read_csv('patience.csv')
pat['Density'] = pat['N']/100
pat['Flow'] = pat['Max']
ax = sns.catplot(x='Density', y='Flow', hue='Pat', data=pat)
ax.fig.suptitle('Flow vs Density, varying Patience, L=100, Range anxiety=3')


# %%
x50 = pd.read_csv('x_varied-50.csv')
x100 = pd.read_csv('x_varied-100.csv')
x300 = pd.read_csv('x_varied-300.csv')
x_varied = pd.read_csv('x_varied.csv')
x400 = pd.read_csv('x_varied-400.csv')
x500 = pd.read_csv('x_varied-500.csv')
plt.scatter(x50['x'], x50['Flow'], marker='.', label='0.5')
plt.scatter(x100['x'], x100['Flow'], marker='.', label='1')
plt.scatter(x_varied['x'], x_varied['Flow'], marker='.', label='2')
plt.scatter(x300['x'], x300['Flow'], marker='.', label='3')
plt.scatter(x400['x'], x400['Flow'], marker='.', label='4')
plt.scatter(x500['x'], x500['Flow'], marker='.', label='5')


plt.legend(title='Density')
plt.xlabel('Ratio of anxious drivers')
plt.ylabel('Flow')
# %%
danger_zone
danger_zone = pd.read_csv('dangerzone.csv')
plt.scatter(danger_zone['N/L'], danger_zone['Flow'], label='3')
plt.xlabel('Density')
plt.ylabel('Flow')

plt.legend(title='Range anxiety')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
r_a[r_a['R_A']==1]
r_a['Density'] = r_a['N']/100
r_a['Flow'] = r_a['Max']
# ax = sns.catplot(x='Density', y='Flow', hue='R_A', data=r_a)
# ax.fig.suptitle('Flow vs Density, varying Range anxiety, L=100, Patience=1')
for i in [0,3,6,9]:
    plt.scatter(r_a[r_a['R_A']==i]['Density'], r_a[r_a['R_A']==i]['Flow'], label=i, marker='.')
plt.legend(title='Range anxiety')
plt.xlabel('Density')
plt.ylabel('Flow')
plt.title('Flow vs Density, varying Range anxiety, L=100, Patience=1')
plt.savefig('images/vary_ra.png', dpi=420)

# %%
pat = pd.read_csv('patience.csv')
pat['Density'] = pat['N']/100
pat['Flow'] = pat['Max']
# ax = sns.catplot(x='Density', y='Flow', hue='Pat', data=pat)
# ax.fig.suptitle('Flow vs Density, varying Patience, L=100, Range anxiety=3')

for i in [0,3,6,9]:
    plt.scatter(pat[pat['Pat']==i]['Density'], pat[pat['Pat']==i]['Flow'], label=i, marker='.')
plt.legend(title='Patience')
plt.xlabel('Density')
plt.ylabel('Flow')
plt.title('Flow vs Density, varying Patience, L=100, Range anxiety=3')
plt.savefig('images/vary_pat.png', dpi=420)


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

plt.title('Flow vs. ratio of anxious drivers varying density, \n patience=4, anxious r_a=9, brave r_a=3')
plt.legend(title='Density')
plt.xlabel('Ratio of anxious drivers')
plt.ylabel('Flow')
plt.savefig('images/anxious.png', dpi=420)

# %%
danger_zone = pd.read_csv('dangerzone.csv')
danger_5 = pd.read_csv('dangerzone_ra5.csv')
danger_0 = pd.read_csv('dangerzone_ra0.csv')
danger_7 = pd.read_csv('dangerzone_ra7.csv')
danger_9 = pd.read_csv('dangerzone_ra9.csv')

plt.scatter(danger_0['N/L'], danger_0['Flow'], label='0', marker='.')
plt.scatter(danger_zone['N/L'], danger_zone['Flow'], label='3', marker='.')
plt.scatter(danger_5['N/L'], danger_5['Flow'], label='5', marker='.')
plt.scatter(danger_7['N/L'], danger_7['Flow'], label='7', marker='.')
plt.scatter(danger_9['N/L'], danger_9['Flow'], label='9', marker='.')

danger_zone
plt.xlabel('Density')
plt.ylabel('Flow')
plt.legend(title='Range anxiety')
plt.title('Fundamental diagram at critical density, \n patience=1, range anxiety=3')
plt.savefig('images/dangerzone.png', dpi=420)

# %%
# RA vs charge
data_with_ra_4 = pd.read_csv('data_with_ra_4.csv')
data_without_ra_4 = pd.read_csv('data_without_ra_4.csv')
data_with_ra_7 = pd.read_csv('data_with_ra_7.csv')
data_without_ra_7 = pd.read_csv('data_without_ra_7.csv')

plt.scatter(data_with_ra_4['N/L'], data_with_ra_4['Flow'], label='10, 4', marker='.')
plt.scatter(data_without_ra_4['N/L'], data_without_ra_4['Flow'], label='6, 0', marker='.')

plt.scatter(data_with_ra_7['N/L'], data_with_ra_7['Flow'], label='10, 7', marker='.')
plt.scatter(data_without_ra_7['N/L'], data_without_ra_7['Flow'], label='3, 0', marker='.')

plt.xlabel('Density')
plt.ylabel('Flow')
plt.title('Fundamental diagram, patience=1')
plt.legend(title='Max charge, range anxiety')

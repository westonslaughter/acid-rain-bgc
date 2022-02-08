#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

hubbard = pd.read_csv('hubbard_chem.csv', index_col=0)

hubbard_wide = pd.pivot_table(hubbard,
                              index=['datetime', 'year', 'month', 'site_code', 'q', 'ms_status', 'ms_interp'],
                              values='val',
                              columns=['var'])

experimental_history = {
    'w1': 'ca_addition',
    'w2': 'timber_operation',
    'w3': 'reference',
    'w4': 'timber_operation',
    'w5': 'timber_operation',
    'w6': 'reference',
    'w7': 'reference',
    'w8': 'reference',
    'w9': 'reference',
}

colors_history = {
    'ca_addition': 'orange',
    'reference': 'blue',
    'timber_operation': 'red',
}

# Hubbard Element Ratios
# Ca:Si
ca_si = hubbard_wide[['Ca', 'SiO2_Si']].reset_index()
ca_si_nona = ca_si.dropna()

# ca_si_nona = ca_si_nona[ca_si_nona['site_code'].str.match("w[0-9]")]

history  = []

for site in ca_si_nona['site_code']:
    history.append(experimental_history[site])

ca_si_nona['history'] = history

# date
# x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in ca_si['datetime']]
# y = ca_si['Ca']
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2000))
# plt.plot(x, y)
# plt.gcf().autofmt_xdate()

# ratio
ca_si_nona['ratio'] = (ca_si_nona['Ca'] / ca_si_nona['SiO2_Si'])
ca_si_nona['log_q'] = np.log(ca_si_nona['q'])

# Ca by SiO2_Si
x = ca_si_nona['SiO2_Si']
y = ca_si_nona['Ca']
plt.scatter(x, y, c=ca_si_nona['history'].map(colors_history) , alpha=0.5)


# Ca:SiO2_Si by Q
fig, ax = plt.subplots()
ax.plot([1, 2])

ax.set_xlabel('Discharge (Q) L/s', fontsize=16)
ax.set_ylabel('Ca:SiO2_Si', fontsize=16)
ax.set_title('Ca:SiO2_Silica Ratio by Discharge', fontsize=16)

x = ca_si_nona['q']
y = ca_si_nona['ratio']
plt.scatter(x, y, alpha=0.5)

# log Q
fig, ax = plt.subplots()
ax.plot([1, 2])

ax.set_xlabel('log(Q) L/s', fontsize=16)
ax.set_ylabel('Ca:SiO2_Si', fontsize=16)
ax.set_title('Ca:SiO2_Silica Ratio by log(Discharge)', fontsize=16)

x = ca_si_nona['log_q']
plt.scatter(x, y, alpha=0.5)

# Ca:Mg
# by Q

# Ca:Na
# by Q

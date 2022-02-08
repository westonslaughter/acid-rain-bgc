#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import datetime as dt

hubbard = pd.read_csv('hbef_elements.csv', index_col=0)

# hubbard_wide = pd.pivot_table(hubbard,
#                               index=['datetime', 'year', 'month', 'site_code', 'q', 'ms_status', 'ms_interp'],
#                               values='val',
#                               columns=['var'])

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
    'timber_operation': 'red',
    'ca_addition': 'orange',
    'reference': 'blue',
}

# helper functions
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

##### Hubbard Element Ratios Dataset Prep
hbef_elements = hubbard[['datetime', 'year', 'month', 'site_code', 'discharge',
                         'Ca', 'SiO2_Si', 'Mg', 'Na', 'Cl', 'K']]
hbef_elements['log_q'] = np.log(hbef_elements['discharge'])

# appending site experimental history as categorical variable
history  = []
for site in hbef_elements['site_code']:
    history.append(experimental_history[site])
hbef_elements['history'] = history

# ratio dfs
casi = hbef_elements[['datetime', 'year', 'month', 'site_code', 'discharge', 'log_q', 'history',
                      'Ca', 'SiO2_Si']].dropna()
camg = hbef_elements[['datetime', 'year', 'month', 'site_code', 'discharge', 'log_q', 'history',
                      'Ca', 'Mg']].dropna()
cana = hbef_elements[['datetime', 'year', 'month', 'site_code', 'discharge', 'log_q', 'history',
                      'Ca', 'Na']].dropna()
# Ca:Si
casi['casi_ratio'] = (casi['Ca'] / casi['SiO2_Si'])
# Ca:Mg
camg['camg_ratio'] = (camg['Ca'] / camg['Mg'])
# Ca:Na
cana['cana_ratio'] = (cana['Ca'] / cana['Na'])

# Ca by SiO2_Si by History
fig, ax = plt.subplots()

ax.set_xlabel('SiO2_Si', fontsize=16)
ax.set_ylabel('Ca', fontsize=16)
ax.set_title('Ca by SiO2_Si in Hubbard Brook', fontsize=16)

for history in colors_history.keys():
    this = casi[casi['history'] == history]
    color = colors_history[history]
    ax.scatter(this['SiO2_Si'], this['Ca'], c=color, label=history, alpha=0.5)

ax.legend(title='Experimental History')

##### Changes in Element Ratios with Flow
#### Calcium and Silica
## Ca:SiO2_Si by Q
fig, ax = plt.subplots()

ax.set_xlabel('Discharge (Q) L/s', fontsize=16)
ax.set_ylabel('Ca:SiO2_Si', fontsize=16)
ax.set_title('Ca:SiO2_Si Ratio by Discharge', fontsize=16)

for history in colors_history.keys():
    this = casi[casi['history'] == history]
    color = colors_history[history]
    ax.scatter(this['discharge'], this['casi_ratio'], c=color, label=history, alpha=0.5)

ax.legend(title="Experimental History")

## Ca:SiO2_Si by log(Q)
# remove outliers from casi_ratio values
casi_z = casi[~is_outlier(casi['casi_ratio'])]
casi_z = casi_z[casi_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

# reorder colors for better image
colors_history = {
    'timber_operation': 'red',
    'reference': 'blue',
    'ca_addition': 'orange',
}

fig, ax = plt.subplots()

ax.set_xlabel('log(Q) L/s', fontsize=16)
ax.set_ylabel('Ca:SiO2_Si', fontsize=16)
plt.suptitle('Ca:SiO2_Si Ratio by log(Q)', fontsize=16)
plt.title('points with Z score greater than 3.5 removed', fontsize=10, y=1)


for history in colors_history.keys():
    this = casi_z[casi_z['history'] == history]
    x = this['log_q']
    y = this['casi_ratio']

    color = colors_history[history]
    ax.scatter(this['log_q'], this['casi_ratio'], c=color, label=history, alpha=0.25)

    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],100)
    ax.plot(X_plot, m*X_plot + b, '-', color=color)

ax.legend(title="Experimental History")

#### Calcium and Magnesium
## Ca:Mg by Q
camg_z = camg_z[camg_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
fig, ax = plt.subplots()

ax.set_xlabel('Discharge (Q) L/s', fontsize=16)
ax.set_ylabel('Ca:Mg', fontsize=16)
ax.set_title('Ca:Mg Ratio by Discharge', fontsize=16)

for history in colors_history.keys():
    this = camg[camg['history'] == history]
    x = this['log_q']
    y = this['camg_ratio']

    color = colors_history[history]
    ax.scatter(this['discharge'], this['camg_ratio'], c=color, label=history, alpha=0.5)

ax.legend(title="Experimental History")

## Ca:Mg Ratio by log(Q)
# remove outliers from camg_ratio values
camg_z = camg[~is_outlier(camg['camg_ratio'], thresh=3)]
# or don't
# camg_z = camg

# reorder colors for better image
colors_history = {
    'reference': 'blue',
    'timber_operation': 'red',
    'ca_addition': 'orange',
}

fig, ax = plt.subplots()

ax.set_xlabel('log(Q) L/s', fontsize=16)
ax.set_ylabel('Ca:Mg', fontsize=16)
plt.suptitle('Ca:Mg Ratio by log(Q)', fontsize=16)
plt.title('points with Z score greater than 3 removed', fontsize=10)

# removing inf values (apparently there are 12?)
camg = camg[camg.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
camg_z = camg_z[camg_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

# adding regression lines
for history in colors_history.keys():
    this = camg_z[camg_z['history'] == history]
    x = this['log_q']
    y = this['camg_ratio']

    color = colors_history[history]
    ax.scatter(x, y, c=color, label=history, alpha=0.25)

    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],100)
    ax.plot(X_plot, m*X_plot + b, '-', color=color)

ax.legend(title="Experimental History")

# Ca:Na
# by Q
fig, ax = plt.subplots()

ax.set_xlabel('Discharge (Q) L/s', fontsize=16)
ax.set_ylabel('Ca:Na', fontsize=16)
ax.set_title('Ca:Na Ratio by Discharge', fontsize=16)

for history in colors_history.keys():
    this = cana[cana['history'] == history]
    color = colors_history[history]
    ax.scatter(this['discharge'], this['cana_ratio'], c=color, label=history, alpha=0.5)

ax.legend(title="Experimental History")

## Ca:Na Ratio by log(Q)
# remove outliers from cana_ratio values
cana_z = cana[~is_outlier(cana['cana_ratio'], thresh=3)]
# or don't
# cana_z = cana

# reorder colors for better image
colors_history = {
    'reference': 'blue',
    'timber_operation': 'red',
    'ca_addition': 'orange',
}

fig, ax = plt.subplots()

ax.set_xlabel('log(Q) L/s', fontsize=16)
ax.set_ylabel('Ca:Na', fontsize=16)
plt.suptitle('Ca:Na Ratio by log(Q)', fontsize=16)
plt.title('points with Z score greater than 3 removed', fontsize=10)

# removing inf values (apparently there are 12?)
cana = cana[cana.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
cana_z = cana_z[cana_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

# adding regression lines
for history in colors_history.keys():
    this = cana_z[cana_z['history'] == history]
    x = this['log_q']
    y = this['cana_ratio']

    # regression stats
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    entry = f'{history}   $r^2$ {r_value:.3f}'

    # scatter
    color = colors_history[history]
    ax.scatter(x, y, c=color, label=entry, alpha=0.25)


    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],100)
    ax.plot(X_plot, m*X_plot + b, '-', color=color)

# overall regression
x = cana_z['log_q']
y = cana_z['cana_ratio']

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
entry = f'overall   $r^2$ {r_value:.3f}'

m, b = np.polyfit(x, y, 1)
X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],100)
ax.plot(X_plot, m*X_plot + b, '--', color='gray', label=entry)

ax.legend(title="Experimental History")









# date
# x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in ca_si['datetime']]
# y = ca_si['Ca']
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2000))
# plt.plot(x, y)
# plt.gcf().autofmt_xdate()

# regression
x = np.array(camg['log_q'].values)
y = np.array(camg['camg_ratio'].values)

gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
mn=np.min(x)
mx=np.max(x)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept

plt.plot(x,y,'ob')
plt.plot(x1,y1,'-r')
plt.show()

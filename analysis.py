#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as grid_spec
import statsmodels.api as sm
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy import stats
import datetime as dt

## Data Prep

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

markers_history = {
    'timber_operation': 's',
    'ca_addition': '^',
    'reference': 'o',
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

# outliers and infs removed
casi_z = casi[~is_outlier(casi['casi_ratio'], thresh=3)]
casi_z = casi_z[casi_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
camg_z = camg[~is_outlier(camg['camg_ratio'], thresh=3)]
camg_z = camg_z[camg_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
cana_z = cana[~is_outlier(cana['cana_ratio'], thresh=3)]
cana_z = cana_z[cana_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

hbef_ratio = hbef_elements[['datetime', 'year', 'month', 'site_code',
                            'discharge', 'Ca', 'SiO2_Si', 'Mg', 'Na',
                            'Cl', 'K']].dropna()

# Ca:Si
hbef_ratio['CaSi'] = (hbef_ratio['Ca'] / hbef_ratio['SiO2_Si'])
# Ca:Mg
hbef_ratio['CaMg'] = (hbef_ratio['Ca'] / hbef_ratio['Mg'])
# Ca:Na
hbef_ratio['CaNa'] = (hbef_ratio['Ca'] / hbef_ratio['Na'])

# outliers and infs removed
hbef_ratio = hbef_ratio[hbef_ratio.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

#  Time Series
hbef_ratio['datetime'] = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in hbef_ratio.datetime]

# determine the water year
hbef_ratio['water_year'] = hbef_ratio.year.where(hbef_ratio.month < 10, hbef_ratio.year + 1)

# annual mean ratios
hbef_wy = hbef_ratio.groupby(['water_year', 'site_code']).mean()
hbef_wy = hbef_wy.reset_index()

#  Hubbard
hbef_exp = hbef_wy[hbef_wy['site_code'].isin(['w2', 'w4', 'w5'])].groupby('water_year').mean().drop('year', 1).reset_index()
hbef_6 = hbef_wy[hbef_wy['site_code'].isin(['w6'])].groupby('water_year').mean().drop('year', 1).reset_index()

# isolating experimental watersheds
hbef_2 = hbef_wy[hbef_wy['site_code'].isin(['w2'])].groupby('water_year').mean().drop('year', 1).reset_index()
hbef_4 = hbef_wy[hbef_wy['site_code'].isin(['w4'])].groupby('water_year').mean().drop('year', 1).reset_index()
hbef_5 = hbef_wy[hbef_wy['site_code'].isin(['w5'])].groupby('water_year').mean().drop('year', 1).reset_index()

#### Plotting

# Ca by SiO2_Si by History
fig, ax = plt.subplots()

ax.set_xlabel('SiO2_Si', fontsize=16)
ax.set_ylabel('Ca', fontsize=16)
ax.set_title('Ca by SiO2_Si in Hubbard Brook', fontsize=16)

# # history color
for history in colors_history.keys():
    this = casi[casi['history'] == history]
    color = colors_history[history]
    ax.scatter(this['SiO2_Si'], this['Ca'], c=color, label=history, alpha=0.5)
#
# # history marker
for history in colors_history.keys():
    this = casi[casi['history'] == history]
    mark = markers_history[history]
    ax.scatter(this['SiO2_Si'], this['Ca'], marker=mark, label=history, alpha=0.5)

# years
years = casi['year'].unique()
years_col = sns.color_palette("flare", len(years))

# years
for index, year in enumerate(years):
    this = casi[casi['year'] == year]
    color = years_col[index]
    m = ax.scatter(this['SiO2_Si'], this['Ca'], c=color, label=year, alpha=0.5)

# plt.colorbar(years_col)
# years + markers
# for index, year in enumerate(years):
#     this = casi[casi['year'] == year]
#     color = years_col[index]
#     histories = this['history'].unique()
#     for index, history in enumerate(histories):
#         this_sub = this[this['history'] == history]
#         mark = markers_history[history]
#         ax.scatter(this['SiO2_Si'], this['Ca'], c=color, label=year, marker=mark, alpha=0.5)

# colorscale
cb = fig.colorbar()
cb.set_label('Color Scale')


ax.legend(title='Experimental History')
ax.legend(title='Years')

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
     # regression stats
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    entry = f'{history}   $r^2$ {r_value:.3f}'

    color = colors_history[history]
    ax.scatter(this['log_q'], this['casi_ratio'], c=color, label=entry, alpha=0.25)

    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],100)
    ax.plot(X_plot, m*X_plot + b, '-', color=color)

# overall regression
x = casi_z['log_q']
y = casi_z['casi_ratio']

radient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ntry = f'overall   $r^2$ {r_value:.3f}'

, b = np.polyfit(x, y, 1)
_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
x.plot(X_plot, m*X_plot + b, '--', color='gray', label=entry)

x.legend(title="Experimental History")

#### Calcium and Magnesium
## Ca:Mg by Q

ig, ax = plt.subplots()

x.set_xlabel('Discharge (Q) L/s', fontsize=16)
x.set_ylabel('Ca:Mg', fontsize=16)
x.set_title('Ca:Mg Ratio by Discharge', fontsize=16)

or history in colors_history.keys():
    this = camg[camg['history'] == history]
    x = this['log_q']
    y = this['camg_ratio']

    color = colors_history[history]
    ax.scatter(this['discharge'], this['camg_ratio'], c=color, label=history, alpha=0.5)

ax.legend(title="Experimental History")

## Ca:Mg Ratio by log(Q)
# remove outliers from camg_ratio values
camg_z = camg[~is_outlier(camg['camg_ratio'], thresh=3)]
camg_z = camg_z[camg_z.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
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

    # regression stats
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    entry = f'{history}   $r^2$ {r_value:.3f}'

    # scatter
    color = colors_history[history]
    ax.scatter(x, y, c=color, label=entry, alpha=0.25)


    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(X_plot, m*X_plot + b, '-', color=color)

# overall regression
x = camg_z['log_q']
y = camg_z['camg_ratio']

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
entry = f'overall   $r^2$ {r_value:.3f}'

m, b = np.polyfit(x, y, 1)
X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
ax.plot(X_plot, m*X_plot + b, '--', color='gray', label=entry)

ax.legend(title="Experimental History")


#### Ca:Na
# Ca:Na by Q
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

#### Facet Grid
# create decade buckets
g = sns.FacetGrid(cana_z, col='site_code', hue='site_code', col_wrap=3, )
g = g.map(plt.plot, 'year', 'Ca')
# g = g.map(plt.fill_between, 'year', 'Ca', alpha=0.2).set_titles("{col_name}")
plt.subplots_adjust(top=0.92)
g = g.fig.suptitle('Calcium Measurements in Hubbard Brook Since 1963')

# Show the graph
plt.show()


#### Ridgelines
sites = [yr for yr in np.unique(cana['site_code'])]
palette = sns.color_palette(None, len(sites))

gs = (grid_spec.GridSpec(len(sites), 1))
fig = plt.figure(figsize=(16, 9))

i = 0

#creating empty list
ax_objs = []

for site_code in sites:
    # creating new axes object and appending to ax_objs
    ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

    # plotting the distribution
    plot = (cana[cana.site_code == site_code]
            .Ca.plot.kde(ax=ax_objs[-1], color="#f0f0f0", lw=0.5)
           )

    # grabbing x and y cana from the kde plot
    x = plot.get_children()[0]._x
    y = plot.get_children()[0]._y

    # filling the space beneath the distribution
    ax_objs[-1].fill_between(x,y, color=palette[i])

    # setting uniform x and y lims
    ax_objs[-1].set_xlim(0, 3)
    ax_objs[-1].set_ylim(0,2.2)

    i += 1

    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax_objs[-1].set_yticklabels([])
    ax_objs[-1].set_ylabel('')

    if i == len(sites)-1:
        pass
    else:
        ax_objs[-1].set_xticklabels([])

    spines = ["top","right","left","bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    site_code = site_code.replace(" ", "\n")
    ax_objs[-1].text(-0.02,0,site_code,fontweight="bold",fontsize=14,ha="center")

plt.tight_layout()
plt.show()

# Time Series

# ca = casi['Ca']
# si = casi['SiO2_Si']

site_colors = {
    'w1':'#6A3EBB',
    'w2':'#286477',
    'w3':'#287746',
    'w4':'#29447A',
    'w5':'#2A447A',
    'w6':'#2A7E57',
    'w7':'#CD8CD9',
    'w8':'#BB8CD9',
    'w9':'#E9D1F0'
}

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2000))

for site in casi.site_code:
    ca = casi[casi.site_code == site]['Ca']
    si = casi[casi.site_code == site]['SiO2_Si']
    x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in casi[casi.site_code == site]['datetime']]

    # Ca
    plt.plot(x, ca, '-', color = site_colors[site])
    # Si
    plt.plot(x, si, '--', color = site_colors[site])

plt.gcf().autofmt_xdate()


ca = casi[casi.site_code == 'w1']['Ca']
si = casi[casi.site_code == 'w1']['SiO2_Si']
casi_ratio =
x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in casi[casi.site_code == 'w1']['datetime']]
# x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in casi[casi.site_code == site]['datetime']]

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2000))

# Ca
plt.plot(x, ca, '-', color = site_colors['w1'])

# Si
plt.plot(x, si, '--', color = site_colors['w1'])
plt.gcf().autofmt_xdate()






#  Hubbard
fig, ax = plt.subplots()
plt.plot(hbef_ratio['datetime'], hbef_ratio.CaSi, linewidth=0.25, color='orange', alpha=.6)
plt.plot(hbef_ratio['datetime'], hbef_ratio.CaNa, linewidth=0.25, color='blue', alpha=.6)
plt.plot(hbef_ratio['datetime'], hbef_ratio.CaMg, linewidth=0.25, color='red', alpha=.6)

fig, ax = plt.subplots()

# Ca:Mg
plt.plot(hbef_exp['water_year'], hbef_exp.CaMg, linewidth=0.5, linestyle='--', color='red', label='Ca:Mg deforested')
plt.plot(hbef_6['water_year'], hbef_6.CaMg, linewidth=0.5, color='red', label='Ca:Mg reference')

# Ca:Si
plt.plot(hbef_6['water_year'], hbef_6.CaSi, linewidth=0.5, color='orange', label='Ca:SiO2-Si')
plt.plot(hbef_exp['water_year'], hbef_exp.CaSi, linewidth=0.5, linestyle='--', color='orange', label='Ca:SiO2-Si')

# Ca:Na
plt.plot(hbef_exp['water_year'], hbef_exp.CaNa, linewidth=0.5, linestyle='--', color='blue', label='Ca:Na')
plt.plot(hbef_6['water_year'], hbef_6.CaNa, linewidth=0.5, color='blue', label='Ca:Na')

# add shaded bar for timber ops
# ax.fill_between(min(1965), max(1967), alpha=0.5)
ax.axvspan(1965, 1967, alpha=0.2)
ax.legend()


# Ca
plt.plot(hbef_exp['water_year'], hbef_exp.Ca, linewidth=0.5, linestyle='--', color='orange', label='Ca deforested')
plt.plot(hbef_exp['water_year'], hbef_exp.Mg, linewidth=0.5, linestyle='--', color='blue', label='Mg deforested')
plt.plot(hbef_6['water_year'], hbef_6.Ca, linewidth=0.5, color='red', label='Ca reference')

# Mg
plt.plot(hbef_6['water_year'], hbef_6.CaMg, linewidth=0.5, color='red', label='Ca:Mg reference')
plt.plot(hbef_6['water_year'], hbef_6.Mg, linewidth=0.5, color='red', linestyle='--',  label='Mg reference')
plt.plot(hbef_6['water_year'], hbef_6.Ca, linewidth=0.5, color='orange', linestyle='--',  label='Ca reference')

# all
fig, ax = plt.subplots()

plt.plot(hbef_6['water_year'], hbef_6.Ca, linewidth=0.5, color='red', label='Ca reference')
plt.plot(hbef_6['water_year'], hbef_6.Mg, linewidth=0.5, color='blue', label='Mg reference')
plt.plot(hbef_6['water_year'], hbef_6.Na, linewidth=0.5, color='green', label='Na reference')
plt.plot(hbef_6['water_year'], hbef_6.K, linewidth=0.5, color='purple', label='K reference')
plt.plot(hbef_6['water_year'], hbef_6.Cl, linewidth=0.5, color='black', label='Cl reference')

ax.axvspan(1965, 1967, alpha=0.2)
ax.legend()

# all
fig, ax = plt.subplots()


# Ratio Time Series

def element_plot(element, element_col='red', y_axis_vals=False):
    """
    plot elements, or ratios, whatever owrks- general plot of experimental
    and reference Hubbard watersheds
    """

    fig, ax = plt.subplots()
    plt.plot(hbef_2['water_year'], hbef_2[element], linewidth=1, linestyle='--', color=element_col, label=f'Deforested \n   Watershed 2')
    # plt.plot(hbef_4['water_year'], hbef_4.element, linewidth=1, linestyle='-.', color='blue')
    plt.plot(hbef_5['water_year'], hbef_5[element], linewidth=1, linestyle='-.', color=element_col, label=f'Watershed 5')
    plt.plot(hbef_6['water_year'], hbef_6[element], linewidth=1, linestyle='-', color='black', label=f'Reference \n   Watershed 6')
    ax.axvspan(1965, 1967, alpha=0.2, color='orange', label = 'W2, devegetated')
    ax.axvspan(1983, 1984, alpha=0.2, color='#FFD580', label='W5, clearcut and herbicide')

    plt.suptitle(f"Hubbard Brook", fontsize=18)
    plt.title(f"{element} from 1963 to 2020, across three watersheds", fontsize=10)

    if y_axis_vals:
        ax.set_ylim(y_axis_vals)

    ax.legend()
    plt.show()

def all_element_plot(watershed, name, y_axis_vals=False, with_q=False):
    fig, ax = plt.subplots()

    plt.plot(watershed['water_year'], watershed.Ca, linewidth=0.5, color='red', label='Ca')
    plt.plot(watershed['water_year'], watershed.Mg, linewidth=0.5, color='blue', label='Mg')
    plt.plot(watershed['water_year'], watershed.Na, linewidth=0.5, color='green', label='Na')
    plt.plot(watershed['water_year'], watershed.K, linewidth=0.5, color='purple', label='K')
    plt.plot(watershed['water_year'], watershed.Cl, linewidth=0.5, color='black', label='Cl')
    plt.suptitle(f"Hubbard Brook {name}", fontsize=18)
    plt.title("concentrations of Ca, Mg, Na, K, and Cl from 1963 to 2020", fontsize=10)

    if name == "Watershed 2":
        ax.axvspan(1965, 1967, alpha=0.2, color='orange', label = 'devegetated')
    elif name == "Watershed 5":
        ax.axvspan(1983, 1984, alpha=0.2, color='#FFD580', label='clearcut and herbicide')
    elif name == "Watershed 6":
        ax.axvspan(1965, 1967, alpha=0.2, color='gray', label = 'W2 devegetated')
        ax.axvspan(1983, 1984, alpha=0.2, color='lightgray', label='W5 clearcut and herbicide')
    if y_axis_vals:
        ax.set_ylim(y_axis_vals)
    if with_q:
        ax2 = ax.twinx()
        ax2.plot(watershed['water_year'], watershed.discharge, linewidth=0.5, color='lightblue', label='discharge')
        ax2.set_ylabel('Discharge L/s')
    ax.set_ylabel('Concentration of Element')
    ax.legend()
    ax2.legend(loc=2)
    plt.show()

# Elements
# all element over time, at one site
all_element_plot(hbef_2, "Watershed 2", y_axis_vals=[0,14], with_q=True)
all_element_plot(hbef_5, "Watershed 5", y_axis_vals=[0,14], with_q=True)
all_element_plot(hbef_6, "Watershed 6", y_axis_vals=[0,14], with_q=True)

# element ratios over time, deforested vs reference
element_plot('CaNa')
element_plot('CaSi')
element_plot('CaMg')

# element concentrations over time, consistent Y max
element_plot('Ca', y_axis_vals=[0, 8])
element_plot('SiO2_Si', y_axis_vals=[0, 8])
element_plot('Mg', y_axis_vals=[0, 8])
element_plot('Na', y_axis_vals=[0, 8])

# Elements by Q

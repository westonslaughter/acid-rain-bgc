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

hubbard = pd.read_csv('hbef_elements_interp.csv', index_col=0)

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

##### Hubbard Element Ratios Dataset
# only small data loss from keeping Cl, K
hbef_ratio = hubbard[['datetime', 'year', 'month', 'site_code', 'q_scaled',
                            'discharge', 'Ca','Ca_flux', 'SiO2_Si', 'SiO2_Si_flux',  'Mg', 'Mg_flux',
                            'Na', 'Na_flux', 'Cl', 'Cl_flux', 'K', 'K_flux']].dropna()


# attach experimental history as a categorical variable
history  = []
for site in hbef_ratio['site_code']:
    history.append(experimental_history[site])
hbef_ratio['history'] = history

# attach decade as a categorical variable
def decader(df, yeartype='year'):
    decade  = []
    for time in df[yeartype]:
        if time < 1970:
            decade.append("1960's")
        elif time < 1980:
            decade.append("1970's")
        elif time < 1990:
            decade.append("1980's")
        elif time < 2000:
            decade.append("1990's")
        elif time < 2010:
            decade.append("2000's")
        elif time < 2020:
            decade.append("2010's")
        elif time < 2030:
            decade.append("2020's")
    df['decade'] = decade


decader(hbef_ratio, 'year')

hbef_ratio['log_q'] = np.log(hbef_ratio['discharge'])
hbef_ratio['scale_log_q'] = np.log(hbef_ratio['q_scaled'])

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
hbef_ratio['water_year'] = hbef_ratio.year.where(hbef_ratio.month < 6, hbef_ratio.year + 1)

# annual mean ratios
hbef_wy = hbef_ratio.groupby(['water_year', 'site_code']).mean()
hbef_wy = hbef_wy.reset_index()

# add in history
history  = []
for site in hbef_wy['site_code']:
    history.append(experimental_history[site])
hbef_wy['history'] = history

#  Hubbard
hbef_exp = hbef_wy[hbef_wy['site_code'].isin(['w2', 'w4', 'w5'])].groupby('water_year').mean().drop('year', 1).reset_index()
hbef_6 = hbef_wy[hbef_wy['site_code'].isin(['w6'])].groupby('water_year').mean().drop('year', 1).reset_index()

# isolating experimental watersheds
hbef_2 = hbef_wy[hbef_wy['site_code'].isin(['w2'])].groupby('water_year').mean().drop('year', 1).reset_index()
hbef_4 = hbef_wy[hbef_wy['site_code'].isin(['w4'])].groupby('water_year').mean().drop('year', 1).reset_index()
hbef_5 = hbef_wy[hbef_wy['site_code'].isin(['w5'])].groupby('water_year').mean().drop('year', 1).reset_index()

#### Plotting
# Ratio Time Series
def element_plot(element, element_name, element_col='red', y_axis_vals=False):
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
    plt.title(f"{element_name} from 1963 to 2020, across three watersheds", fontsize=10)

    if y_axis_vals:
        ax.set_ylim(y_axis_vals)

    ax.set_ylabel(f'{element_name}')
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
all_element_plot(hbef_2, "Watershed 2", with_q=True)
all_element_plot(hbef_5, "Watershed 5", with_q=True)
all_element_plot(hbef_6, "Watershed 6", with_q=True)

# element ratios over time, deforested vs reference
element_plot('CaNa', 'Ca:Na')
element_plot('CaSi', 'Ca:Si')
element_plot('CaMg', 'Ca:Mg')

# element concentrations over time, consistent Y max
element_plot('Ca', '[Ca]', y_axis_vals=[0, 8])
element_plot('SiO2_Si', '[SiO2_Si]', y_axis_vals=[0, 8])
element_plot('Mg', '[Mg]', y_axis_vals=[0, 8])
element_plot('Na', '[Na]', y_axis_vals=[0, 8])

# Elements by Q
# Ca by SiO2_Si by History
def variable_scatter(df, element_x, element_y, var,
                     log_x=False, log_y=False, regress=False,
                     title_add=""):
    fig, ax = plt.subplots()

    x_col = f'{element_x}'
    df[x_col] = df[element_x]
    y_col = f'{element_y}'
    df[y_col] = df[element_y]

    if log_x is True:
        x_col = f'log_{element_x}'
        df[x_col] = np.log(df[element_x])

    if log_y is True:
        y_col = f'log_{element_y}'
        df[y_col] = np.log(df[element_y])

    ax.set_xlabel(element_x, fontsize=16)
    ax.set_ylabel(element_y, fontsize=16)
    ax.set_title(f'{element_x} by {element_y} in Hubbard Brook\n{title_add}', fontsize=16)

    # var
    if var:
        var_list = df[var].unique()
        var_list_col = sns.color_palette("flare", len(var_list))

        # var_list
        for index, item in enumerate(var_list):
            this = df[df[var] == item]
            color = var_list_col[index]

            if regress:
                x = this[x_col]
                y = this[y_col]

                # regression stats
                gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                entry = f'{item}   $r^2$ {r_value:.3f}'
                ax.scatter(x, y, c=color, label=entry, alpha=0.25)

                m, b = np.polyfit(x, y, 1)
                X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
                ax.plot(X_plot, m*X_plot + b, '-', color=color)
            else:
                m = ax.scatter(this[x_col], this[y_col], c=color, label=item, alpha=0.5)
    else:
         m = ax.scatter(df[x_col], df[y_col], alpha=0.5)

    if log_x is True:
        ax.set_xlabel(f'log({element_x})', fontsize=16)
    if log_y is True:
        ax.set_ylabel(f'log({element_y})', fontsize=16)

    # overall regression
    x = df[x_col]
    y = df[y_col]

    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    entry = f'overall   $r^2$ {r_value:.3f}'

    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(X_plot, m*X_plot + b, '--', color='gray', label=entry)

    if var:
        ax.legend(title=f'{var}')
    else:
        ax.legend()
    # plt.show()


# Scatter & Regression
def history_scatter(df, ratio, z_thresh=False):
    if z_thresh:
        df = df[~is_outlier(df[ratio], thresh=z_thresh)]

    df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    # reorder colors for better image
    colors_history = {
        'reference': 'blue',
        'timber_operation': 'red',
        'ca_addition': 'orange',
    }

    fig, ax = plt.subplots()

    ax.set_xlabel('log(Q) L/s', fontsize=16)
    ax.set_ylabel(f'{ratio}', fontsize=16)
    plt.suptitle(f'{ratio} Ratio by log(Q)', fontsize=16)
    plt.title('points with Z score greater than 3 removed', fontsize=10)

    # adding regression lines
    for history in colors_history.keys():
        this = df[df['history'] == history]
        x = this['log_q']
        y = this[ratio]

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
    x = df['log_q']
    y = df[ratio]

    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    entry = f'overall   $r^2$ {r_value:.3f}'

    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(X_plot, m*X_plot + b, '--', color='gray', label=entry)
    ax.legend(title="Experimental History")

    plt.show()

# scatter by history
history_scatter(hbef_ratio, 'CaMg', z_thresh=3)
history_scatter(hbef_ratio, 'CaSi', z_thresh=3)
history_scatter(hbef_ratio, 'CaNa', z_thresh=3)

# scatter by year
# all sites
variable_scatter(hbef_ratio, 'Ca', 'SiO2_Si', 'decade', log_x=True, log_y=True, regress=True)
variable_scatter(hbef_ratio, 'Ca', 'Mg', 'decade', log_x=True, log_y=True, regress=True)
variable_scatter(hbef_ratio, 'Ca', 'Na', 'decade', log_x=True, log_y=True, regress=True)

# deforested
# make 'water decades' using annual means data
# W5
decader(hbef_5, 'water_year')
variable_scatter(hbef_5, 'Ca', 'SiO2_Si', 'decade', log_x=True, log_y=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_5, 'Ca', 'Mg', 'decade', log_x=True, log_y=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_5, 'Ca', 'Na', 'decade', log_x=True, log_y=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")


variable_scatter(hbef_5, 'discharge', 'CaSi', 'decade', title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_5, 'discharge', 'CaMg', 'decade', title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_5, 'discharge', 'CaNa', 'decade', title_add="Watershed 5, clearcut and herbicide 1983-1984")

# W2
decader(hbef_2, 'water_year')
variable_scatter(hbef_2, 'Ca', 'SiO2_Si', 'decade', log_x=True, log_y=True, title_add="Watershed 2, devegetated 1965-1967")
variable_scatter(hbef_2, 'Ca', 'Mg', 'decade', log_x=True, log_y=True, title_add="Watershed 2, devegetated 1965-1967")
variable_scatter(hbef_2, 'Ca', 'Na', 'decade', log_x=True, log_y=True, title_add="Watershed 2, devegetated 1965-1967")


variable_scatter(hbef_2, 'discharge', 'CaSi', 'decade', title_add="Watershed 2, devegetated 1965-1967")
variable_scatter(hbef_2, 'discharge', 'CaMg', 'decade', title_add="Watershed 2, devegetated 1965-1967")
variable_scatter(hbef_2, 'discharge', 'CaNa', 'decade', title_add="Watershed 2, devegetated 1965-1967")

# W6
decader(hbef_6, 'water_year')
variable_scatter(hbef_6, 'Ca', 'SiO2_Si', 'decade', log_x=True, log_y=True, title_add="Reference Watershed")
variable_scatter(hbef_6, 'Ca', 'Mg', 'decade', log_x=True, log_y=True, title_add="Reference Watershed")
variable_scatter(hbef_6, 'Ca', 'Na', 'decade', log_x=True, log_y=True, title_add="Reference Watershed")


variable_scatter(hbef_6, 'discharge', 'CaSi', 'decade', title_add="Reference Watershed")
variable_scatter(hbef_6, 'discharge', 'CaMg', 'decade', title_add="Reference Watershed")
variable_scatter(hbef_6, 'discharge', 'CaNa', 'decade', title_add="Reference Watershed")

# using all data points
# W5
variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'Ca', 'SiO2_Si', 'decade', log_x=True, regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'Ca', 'Mg', 'decade', regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'Ca', 'Na', 'decade', regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")

variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'discharge', 'SiO2_Si', 'decade', log_x=True, regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'discharge', 'Mg', 'decade',  log_x=True, regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'discharge', 'Na', 'decade',  log_x=True, regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")

variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'discharge', 'CaSi', 'decade', log_x=True, log_y=True, regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'discharge', 'CaMg', 'decade',  log_x=True, regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")
variable_scatter(hbef_ratio[hbef_ratio['site_code'].isin(['w5'])], 'discharge', 'CaNa', 'decade',  log_x=True, regress=True, title_add="Watershed 5, clearcut and herbicide 1983-1984")


# megafacet
# element by flow across elements
hbef = hbef_ratio[hbef_ratio['site_code'].isin(['w5', 'w2', 'w6'])]
cols = hbef.site_code.value_counts().shape[0]


elements = ['Ca', 'Na', 'Mg']
rows = len(elements)

fig, ax = plt.subplots(rows, cols, figsize=(10, 10))

for index, element in enumerate(elements):
    for i, site in enumerate(hbef.site_code.value_counts().index.values):

        site_label = {
            'w6': '${reference}$',
            'w2': '${devegetated, 1965-67}$',
            'w5': '${clear cut, 1983-84}$'
        }

        this_label = site_label[site]
        site_bold = f"$\\bf{site}$"

        df = hbef[hbef['site_code'] == site]

        x_col = 'scale_log_q'
        y_col = element

        ax[index][i].axes.yaxis.set_visible(False)
        ax[index][i].axes.xaxis.set_visible(False)

        if index ==0:
            ax[index][i].set_title(f'{site_bold}\n{this_label}', fontsize=10, y=1)
        if i == 0:
            # concentration
            ax[index][i].set_ylabel(f'[{y_col}]', fontsize=12)

            ax[index][i].axes.yaxis.set_visible(True)
        if index == len(elements) -1 and i == 1:
            ax[index][i].set_xlabel("$log(Q)_{scaled}$", fontsize = 12)


        ax[index][i].set_ylim([-0.25, 6])

        var_list = df["decade"].unique()
        var_list_col = sns.color_palette("flare", len(var_list))

        # var_list
        for var_index, item in enumerate(var_list):
            this = df[df["decade"] == item]
            color = var_list_col[var_index]

            x = this[x_col]
            y = this[y_col]

            # regression stats
            gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            entry = f'{item}   $r^2$ {r_value:.3f}'
            im1 = ax[index][i].scatter(x, y, c=color, label=entry, alpha=0.25)

            m, b = np.polyfit(x, y, 1)
            X_plot = np.linspace(ax[index][i].get_xlim()[0], ax[index][i].get_xlim()[1], 100)
            ax[index][i].plot(X_plot, m*X_plot + b, '-', color=color)

        if index == len(elements)-1:
            ax[index][i].axes.xaxis.set_visible(True)

norm = plt.Normalize(hbef.year.min(), hbef.year.max())
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
fig.suptitle("Element Concentration by Site in Hubbard Brook\n1963-2020, interpolated dataset", x=0.42, y=.98)
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.9, aspect=38)
plt.show()


# element flux
fluxes = ['Ca_flux', 'Na_flux', 'Mg_flux', 'Cl_flux', 'K_flux']
rows = len(fluxes)

fig, ax = plt.subplots(rows, cols, figsize=(8, 8))

for index, flux in enumerate(fluxes):
    for i, site in enumerate(hbef.site_code.value_counts().index.values):
        df = hbef[hbef['site_code'] == site]

        site_label = {
            'w6': '${reference}$',
            'w2': '${devegetated, 1965-67}$',
            'w5': '${clear cut, 1983-84}$'
        }

        this_label = site_label[site]
        site_bold = f"$\\bf{site}$"

        x_col = 'q_scaled'
        y_col = flux
        y_name = y_col.split('_')[0]

        ax[index][i].axes.yaxis.set_visible(False)
        ax[index][i].axes.xaxis.set_visible(False)

        if index ==0:
            ax[index][i].set_title(f'{site_bold}\n{this_label}', fontsize=10, y=1)
        if i == 0:
            ax[index][i].set_ylabel(f'${y_name}$ ', fontsize=12)

            ax[index][i].axes.yaxis.set_visible(True)
        if index == len(fluxes) -1 and i == 1:
            ax[index][i].set_xlabel("$Q_{scaled}$", fontsize = 12)

        # ax[index][i].set_ylim([-0.25, 6])

        var_list = df["decade"].unique()
        var_list_col = sns.color_palette("flare", len(var_list))

        # var_list
        for var_index, item in enumerate(var_list):
            this = df[df["decade"] == item]
            color = var_list_col[var_index]

            x = this[x_col]
            y = this[y_col]

            # regression stats
            gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            entry = f'{item}   $r^2$ {r_value:.3f}'
            im1 = ax[index][i].scatter(x, y, c=color, label=entry, alpha=0.25)

            m, b = np.polyfit(x, y, 1)
            X_plot = np.linspace(ax[index][i].get_xlim()[0], ax[index][i].get_xlim()[1], 100)
            ax[index][i].plot(X_plot, m*X_plot + b, '-', color=color)

        if index == len(fluxes)-1:
            ax[index][i].axes.xaxis.set_visible(True)

norm = plt.Normalize(hbef.year.min(), hbef.year.max())
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
fig.suptitle("Site Element Flux by Watershed Scaled Discharge in Hubbard Brook", x=0.42, y=.98)
# fig.text("${1963-2020}$, interpolated data included", s=0.42, y=.96)
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.9, aspect=38)
plt.show()

# element:ratios by flow across elements
element_ratio_ratios = ['CaMg', 'CaNa', 'CaMg']
rows = len(element_ratio_ratios)

fig, ax = plt.subplots(rows, cols, figsize=(10, 10))

for index, element_ratio in enumerate(element_ratio_ratios):
    for i, site in enumerate(hbef.site_code.value_counts().index.values):

        site_label = {
            'w6': '${reference}$',
            'w2': '${devegetated, 1965-67}$',
            'w5': '${clear cut, 1983-84}$'
        }

        this_label = site_label[site]
        site_bold = f"$\\bf{site}$"

        df = hbef[hbef['site_code'] == site]

        x_col = 'scale_log_q'
        y_col = element_ratio

        ax[index][i].axes.yaxis.set_visible(False)
        ax[index][i].axes.xaxis.set_visible(False)

        if index ==0:
            ax[index][i].set_title(f'{site_bold}\n{this_label}', fontsize=10, y=1)
        if i == 0:
            # ratio
            ratio_string = y_col[:2] + ':' + y_col[2:]
            ax[index][i].set_ylabel(f'{ratio_string}', fontsize=12)

            ax[index][i].axes.yaxis.set_visible(True)
        if index == len(element_ratio_ratios) -1 and i == 1:
            ax[index][i].set_xlabel("$log(Q)_{scaled}$", fontsize = 12)

        # y lim
        # ax[index][i].set_ylim([-0.25, 6])

        var_list = df["decade"].unique()
        var_list_col = sns.color_palette("flare", len(var_list))

        # var_list
        for var_index, item in enumerate(var_list):
            this = df[df["decade"] == item]
            color = var_list_col[var_index]

            x = this[x_col]
            y = this[y_col]

            # regression stats
            gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            entry = f'{item}   $r^2$ {r_value:.3f}'
            im1 = ax[index][i].scatter(x, y, c=color, label=entry, alpha=0.25)

            m, b = np.polyfit(x, y, 1)
            X_plot = np.linspace(ax[index][i].get_xlim()[0], ax[index][i].get_xlim()[1], 100)
            ax[index][i].plot(X_plot, m*X_plot + b, '-', color=color)

        if index == len(element_ratio_ratios)-1:
            ax[index][i].axes.xaxis.set_visible(True)

norm = plt.Normalize(hbef.year.min(), hbef.year.max())
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
fig.suptitle("Calcium:Element Ratios by Site in Hubbard Brook\n1963-2020, interpolated dataset", x=0.42, y=.98)
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.9, aspect=38)
plt.show()

# Time Series
# annual flux time series,
hbef_yield = hbef.groupby(['water_year', 'site_code']).sum().reset_index()
# and, annual difference in yield for each element
# calculate difference w2-ref yield, w5-ref yield
hbef_yield['Ca_ref'] = 0
hbef_yield.set_index('water_year', 'site_code')

calcium_ref = {
    'w5': [],
    'w2': []
}

# for element in fluxes:
ca_ref = []
for element in fluxes:
    element_ref_list = []
    for index, row in hbef_yield.iterrows():
        if row['site_code'] == 'w5' or row['site_code'] == 'w2':
            year = row['water_year']
            site = row['site_code']
            val = row[element]
            ref = hbef_yield[hbef_yield['water_year'] == year][hbef_yield['site_code']== 'w6'][element].values[0]
            ca_ref_val = val - ref
            element_ref_list.append(ca_ref_val)
        # ca_ref.append([year, site, ca_ref_val])
        else:
            element_ref_list.append(0)

    ele_col = element+'_ref'
    hbef_yield[ele_col] = element_ref_list





# and, total record difference in yield for each element

# facet of timeseries
# this shows a timersies of annual flux sums, with a
# gray line showing the difference between experimental annual flux and reference flux
#
fluxes = ['Ca_flux', 'Na_flux', 'Mg_flux', 'Cl_flux', 'K_flux']
cols = hbef_yield.site_code.value_counts().shape[0]
rows = len(fluxes)



site_col = {
    'w5':['-', 'red'],
    'w2':['-', 'red'],
    'w6':['-', 'blue']
}

sites = ['w6', 'w5', 'w2']

fig, ax = plt.subplots(rows, cols, figsize=(10, 10))

for index, element_yield in enumerate(fluxes):
    for i, site in enumerate(sites):
        x = hbef_yield[hbef_yield['site_code']==site]['water_year']
        y = hbef_yield[hbef_yield['site_code']==site][element_yield]
        ax[index][i].plot(x, y, linewidth=1, color=site_col[site][1], linestyle=site_col[site][0], label=f'{site} Annual {element_yield}')

        site_bold = f"$\\bf{site}$"

        ax[index][i].axes.yaxis.set_visible(False)
        ax[index][i].axes.xaxis.set_visible(False)


        if index == 0:
            ax[index][i].set_title(f'{site_bold}', fontsize=12, y=1)
        if i == 0:
            element_string = element_yield.split('_')[0]
            ax[index][i].set_ylabel(f'{element_string}', fontsize=12)
            ax[index][i].axes.yaxis.set_visible(True)

        fluxref = element_yield+'_ref'

        if site == 'w2':
            ref = hbef_yield[hbef_yield['site_code']==site][fluxref]
            ax[index][i].axvspan(1965, 1967, alpha=0.2,
                                 color='gray',
                                 label='W2 devegetated')
            ax[index][i].plot(x, ref, linewidth=1, color='gray',
                              linestyle=site_col[site][0],
                              label=f'difference annual yield, {site} and reference',
                              alpha=0.5)
        elif site=='w5':
            ref = hbef_yield[hbef_yield['site_code']==site][fluxref]
            ax[index][i].axvspan(1983, 1984, alpha=0.2, color='lightgray', label='W5 clearcut and herbicide')
            ax[index][i].plot(x, ref, linewidth=1, color='gray',
                              linestyle=site_col[site][0],
                              label=f'difference annual yield, {site} and reference',
                              alpha=0.5)

        # ax[index][i].set_ylim([0, 3000])
        if index == 0:
            ax[index][i].set_ylim([0, 80])
        elif index == 1:
            ax[index][i].set_ylim([0, 20])
        elif index == 2:
            ax[index][i].set_ylim([0, 20])
        elif index == 3:
            ax[index][i].set_ylim([0, 15])
        elif index == 4:
            ax[index][i].set_ylim([0, 40])


        if index == len(fluxes)-1:
            ax[index][i].axes.xaxis.set_visible(True)

fig.suptitle("Element Annual Yields Over Time in Hubbard Brook\n1963-2020, interpolated dataset", x=0.42, y=.98)
from matplotlib.lines import Line2D
colors = ['blue', 'red']
lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='-') for c in colors]
labels = ['W6, biogeochemical reference', 'W2 and W5, experimental']
fig.legend(lines, labels, loc='upper right')
plt.show()

# now we explore 'total fluxes'
# we will sum all annual fluxes, and display a pie chart of sires with a bubble
# and number sized to total export since 1963
# cold do a nested pie chart with each circle being a decade, and the proportions of export
# of each site shown, with total export values
decader(hbef_yield, 'water_year')

hbef_decade = hbef_decade.reset_index()
fig1, ax1 = plt.subplots()


x = hbef_decade[hbef_decade['site_code']=='w5']['decade']
y = hbef_decade[hbef_decade['site_code']=='w5']['Ca_flux']

ax1.pie(y,labels=x, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# stacked bar chart
# plot bars in stack manner
x = hbef_decade[hbef_decade['site_code']=='w5']['decade'].unique()
hbef_decs = hbef_decade[['decade', 'site_code', 'Ca_flux']]

df1 = (hbef_decs.pivot_table(index='site_code',
                      columns='decade',
                      values='Ca_flux'))

df1.loc[:,["1960's", "1970's", "1980's", "1990's", "2000's", "2010's", "2020's"]].plot.bar(stacked=True)

#
fig, ax = plt.subplots()

ax.bar(df1.index, df1["1960's"], colors=var_list_col)

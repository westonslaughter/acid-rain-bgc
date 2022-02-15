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

# preparation
## design
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

since_harvest = {
    'w2':[1965, 1966, 1967],
    'w5':[1984, 1985]
    ,}

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



## helper functions
def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

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

def since_harvester(df, site, yeartype):
    since_harvest = []
    yr = yeartype
    if site == 'w2':
        for index, row in df.iterrows():
            if row[yr] < 1965:
                since_harvest.append('reference')
            elif row[yr] < 1975:
                since_harvest.append(0)
            elif row[yr] < 1985:
                since_harvest.append(1)
            elif row[yr] < 1995:
                since_harvest.append(2)
            elif row[yr] < 2005:
                since_harvest.append(3)
            elif row[yr] < 2015:
                since_harvest.append(4)
            elif row[yr] < 2025:
                since_harvest.append(5)
            else:
                print(row[yr])
    elif site=='w5':
        for index, row in df.iterrows():
            if row[yr] < 1963:
                since_harvest.append('reference')
            elif row[yr] < 1973:
                since_harvest.append('reference')
            elif row[yr] < 1983:
                since_harvest.append('reference')
            elif row[yr] < 1993:
                since_harvest.append(0)
            elif row[yr] < 2003:
                since_harvest.append(1)
            elif row[yr] < 2013:
                since_harvest.append(2)
            elif row[yr] < 2023:
                since_harvest.append(3)
            else:
                print(row[yr])
    else:
        for index, row in df.iterrows():
            since_harvest.append('reference')

    df['since'] = since_harvest

## data munge
hubbard = pd.read_csv('hbef_elements_interp.csv', index_col=0)
hbef_ratio = hubbard[['datetime', 'year', 'month', 'site_code', 'q_scaled',
                            'discharge', 'Ca','Ca_flux', 'SiO2_Si', 'SiO2_Si_flux',  'Mg', 'Mg_flux',
                            'Na', 'Na_flux', 'Cl', 'Cl_flux', 'K', 'K_flux']].dropna()
# adding derived variables
# history
history  = []
for site in hbef_ratio['site_code']:
    history.append(experimental_history[site])
hbef_ratio['history'] = history
decader(hbef_ratio, 'year')

# log q
hbef_ratio['log_q'] = np.log(hbef_ratio['discharge'])
# area scaled log q
hbef_ratio['scale_log_q'] = np.log(hbef_ratio['q_scaled'])
# Ca:Si
hbef_ratio['CaSi'] = (hbef_ratio['Ca'] / hbef_ratio['SiO2_Si'])
# Ca:Mg
hbef_ratio['CaMg'] = (hbef_ratio['Ca'] / hbef_ratio['Mg'])
# Ca:Na
hbef_ratio['CaNa'] = (hbef_ratio['Ca'] / hbef_ratio['Na'])
#  Time Series
hbef_ratio['datetime'] = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in hbef_ratio.datetime]
# determine the water year
hbef_ratio['water_year'] = hbef_ratio.year.where(hbef_ratio.month < 6, hbef_ratio.year + 1)
# outliers and infs removed
hbef_ratio = hbef_ratio[hbef_ratio.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

# annual mean
hbef_wy_mean = hbef_ratio.groupby(['water_year', 'site_code']).mean().reset_index()

# annual sum
hbef_wy_sum = hbef_ratio.groupby(['water_year', 'site_code']).sum().reset_index()

# all-record
hbef_ratio = hbef_ratio

# single sites
# all-record
hbef_w6 = hbef_ratio[hbef_ratio['site_code']=='w6']
hbef_w2 = hbef_ratio[hbef_ratio['site_code']=='w2']
hbef_w5 = hbef_ratio[hbef_ratio['site_code']=='w5']
# annual
#       mean
hbef_w6_wy_mean = hbef_wy_mean[hbef_wy_mean['site_code']=='w6']
hbef_w2_wy_mean = hbef_wy_mean[hbef_wy_mean['site_code']=='w2']
hbef_w5_wy_mean = hbef_wy_mean[hbef_wy_mean['site_code']=='w5']
#       sum
hbef_w6_wy_sum = hbef_wy_sum[hbef_wy_sum['site_code']=='w6']
hbef_w2_wy_sum = hbef_wy_sum[hbef_wy_sum['site_code']=='w2']
hbef_w5_wy_sum = hbef_wy_sum[hbef_wy_sum['site_code']=='w5']

# getting decades since harvest
# all-record
since_harvester(hbef_w5, 'w5', 'year')
since_harvester(hbef_w2, 'w2', 'year')
since_harvester(hbef_w6, 'w6', 'year')
hbef_ratio = hbef_w6.append(hbef_w2).append(hbef_w5)

# annual
# mean
since_harvester(hbef_w5_wy_mean, 'w5', 'water_year')
since_harvester(hbef_w2_wy_mean, 'w2', 'water_year')
since_harvester(hbef_w6_wy_mean, 'w6', 'water_year')
hbef_wy_mean = hbef_w6_wy_mean.append(hbef_w2_wy_mean).append(hbef_w5_wy_mean)

# sum
since_harvester(hbef_w5_wy_sum, 'w5', 'water_year')
since_harvester(hbef_w2_wy_sum, 'w2', 'water_year')
since_harvester(hbef_w6_wy_sum, 'w6', 'water_year')
hbef_wy_sum = hbef_w6_wy_sum.append(hbef_w2_wy_sum).append(hbef_w5_wy_sum)



#### Questions
### are there changes in Ca:Si, Ca:Mg, Ca:Na ratios with flow?
# BELOW: facets of scatter plot C:Q, flux:Q, and ratio:Q
# ALL RECORDS (inrerpolated)
# element by flow across elements
hbef = hbef_ratio[hbef_ratio['site_code'].isin(['w5', 'w2', 'w6'])]
elements = ['Ca', 'SiO2_Si', 'Na', 'Mg', 'K']

cols = hbef.site_code.value_counts().shape[0]
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
element_ratio_ratios = ['CaMg', 'CaSi', 'CaNa', 'CaMg']
rows = len(element_ratio_ratios)

fig, ax = plt.subplots(rows, cols, figsize=(10, 10))

var_list = hbef["since"].unique()
var_list_col = sns.color_palette("flare", len(var_list))
var_list_col[0] = 'lightblue'


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
        ax[index][i].set_ylim([0, 10])

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


## ratios && ratio:Q relationships
#       differ between deforested and forested watersheds?
#       changed over time?


### look at [element]:Q and Ratio:Q relationships
#       by decades since harvest

# ALL RECORDS, but, colored by 'decades since harvest'
# element by flow across elements
fluxes = ['Ca_flux', 'Na_flux', 'Mg_flux', 'Cl_flux', 'K_flux']
rows = len(fluxes)
fig, ax = plt.subplots(rows, cols, figsize=(8, 8))
var_list = hbef["since"].unique()
var_list_col = sns.color_palette("flare", len(var_list))
var_list_col[0] = 'lightblue'

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

        this_var_list = df["since"].unique()
        this_var_list = [item for item in range(len(this_var_list)-1)]
        this_var_list.insert(0, 'reference')
        this_var_list = np.array(this_var_list, dtype='object')

        # var_list
        for var_index, item in enumerate(this_var_list):
            if item in var_list:
                col_index = np.where(var_list == item)[0][0]
            else:
                col_index=0
                print('var not in color list')

            this = df[df["since"] == item]
            color = var_list_col[col_index]

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

fig.suptitle("Site Element Flux by Watershed Scaled Discharge in Hubbard Brook\n colored by decades since harvest", x=0.42, y=.98)
plt.legend()
plt.show()

# element ratios
element_ratios = ['CaMg', 'CaNa', 'CaSi']
rows = len(element_ratios)

fig, ax = plt.subplots(rows, cols, figsize=(8, 8))

var_list = hbef["since"].unique()
var_list_col = sns.color_palette("flare", len(var_list))
var_list_col[0] = 'lightblue'

for index, flux in enumerate(element_ratios):
    for i, site in enumerate(hbef.site_code.value_counts().index.values):
        df = hbef[hbef['site_code'] == site]

        site_label = {
            'w6': '${reference}$',
            'w2': '${devegetated, 1965-67}$',
            'w5': '${clear cut, 1983-84}$'
        }

        this_label = site_label[site]
        site_bold = f"$\\bf{site}$"

        x_col = 'scale_log_q'
        y_col = flux
        y_name = y_col[:2] + ':' + y_col[2:]

        ax[index][i].axes.yaxis.set_visible(False)
        ax[index][i].axes.xaxis.set_visible(False)

        if index ==0:
            ax[index][i].set_title(f'{site_bold}\n{this_label}', fontsize=10, y=1)
        if i == 0:
            ax[index][i].set_ylabel(f'${y_name}$ ', fontsize=12)

            ax[index][i].axes.yaxis.set_visible(True)
        if index == len(element_ratios) -1 and i == 1:
            ax[index][i].set_xlabel("$log(Q)_{scaled}$", fontsize = 12)

        ax[index][i].set_ylim([0, 10])

        this_var_list = df["since"].unique()
        this_var_list = [item for item in range(len(this_var_list)-1)]
        this_var_list.insert(0, 'reference')
        this_var_list = np.array(this_var_list, dtype='object')

        # var_list
        for var_index, item in enumerate(this_var_list):
            if item in var_list:
                col_index = np.where(var_list == item)[0][0]
            else:
                col_index=0
                print('var not in color list')

            this = df[df["since"] == item]
            color = var_list_col[col_index]

            x = this[x_col]
            y = this[y_col]

            # regression stats
            gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            entry = f'{item}   $r^2$ {r_value:.3f}'
            im1 = ax[index][i].scatter(x, y, c=color, label=entry, alpha=0.25)

            m, b = np.polyfit(x, y, 1)
            X_plot = np.linspace(ax[index][i].get_xlim()[0], ax[index][i].get_xlim()[1], 100)
            ax[index][i].plot(X_plot, m*X_plot + b, '-', color=color)

        if index == len(element_ratios)-1:
            ax[index][i].axes.xaxis.set_visible(True)

fig.suptitle("Site Element Flux by Watershed Scaled Discharge in Hubbard Brook\n colored by decades since harvest", x=0.42, y=.98)
plt.legend()
plt.show()

# element concentrations
elements = ['Ca', 'Mg', 'SiO2_Si', 'K', 'Cl']
rows = len(elements)

fig, ax = plt.subplots(rows, cols, figsize=(8, 8))

var_list = hbef["since"].unique()
var_list_col = sns.color_palette("flare", len(var_list))
var_list_col[0] = 'lightblue'

for index, element in enumerate(elements):
    for i, site in enumerate(hbef.site_code.value_counts().index.values):
        df = hbef[hbef['site_code'] == site]

        site_label = {
            'w6': '${reference}$',
            'w2': '${devegetated, 1965-67}$',
            'w5': '${clear cut, 1983-84}$'
        }

        this_label = site_label[site]
        site_bold = f"$\\bf{site}$"

        x_col = 'scale_log_q'
        y_col = element
        y_name = '[' + y_col + ']'

        ax[index][i].axes.yaxis.set_visible(False)
        ax[index][i].axes.xaxis.set_visible(False)

        if index ==0:
            ax[index][i].set_title(f'{site_bold}\n{this_label}', fontsize=10, y=1)
        if i == 0:
            ax[index][i].set_ylabel(f'${y_name}$ ', fontsize=12)

            ax[index][i].axes.yaxis.set_visible(True)
        if index == len(elements) -1 and i == 1:
            ax[index][i].set_xlabel("$log(Q)_{scaled}$", fontsize = 12)

        ax[index][i].set_ylim([0, 10])

        this_var_list = df["since"].unique()
        this_var_list = [item for item in range(len(this_var_list)-1)]
        this_var_list.insert(0, 'reference')
        this_var_list = np.array(this_var_list, dtype='object')

        # var_list
        for var_index, item in enumerate(this_var_list):
            if item in var_list:
                col_index = np.where(var_list == item)[0][0]
            else:
                col_index=0
                print('var not in color list')

            this = df[df["since"] == item]
            color = var_list_col[col_index]

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

fig.suptitle("Site Element Concentration by Watershed Scaled Discharge in Hubbard Brook\n colored by decades since harvest", x=0.42, y=.98)
plt.legend()
plt.show()

### calculate the total difference in yield
##       between the deforested watersheds and watershed 6
#               by year
# stacked bar chart (individual)
def element_stack(sum_data, element, alt_var="decade", pivot=True, drop_index=1):
    if pivot == True:
        decader(sum_data, 'water_year')
        sum_data = (sum_data.pivot_table(index=alt_var,
                      columns='site_code',
                      values=element))

    # Get some pastel shades for the colors
    data = sum_data.values[drop_index:]
    columns = ('w2', 'w5', 'w6')
    rows = sum_data.index[drop_index:]
    colors = plt.cm.BuPu(np.linspace(0.1, .9, len(rows)))
    n_rows = len(data)


    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.2f' % x for x in y_offset])

    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')
    the_table.scale(1,2)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.5, bottom=0.5)

    plt.ylabel(element)
    plt.xticks([])
    plt.title('Flux by Site\n colored by decade')
    # plt.show()

element_stack(hbef_wy_mean, 'SiO2_Si_flux', alt_var='decade', pivot=True)

# overall total yireld w6 vs w2 and w5
# bar chart
wy_ca_sum = (hbef_wy_sum.pivot_table(index="site_code",
                      columns='water_year',
                      values="Ca_flux"))

wy_ca_sum['ca_total'] = wy_ca_sum.sum(axis=1)
wy_ca_sum = wy_ca_sum[['ca_total']]

# Get some pastel shades for the colors
data = wy_ca_sum.values
columns = ('w2', 'w5', 'w6')
rows = wy_ca_sum.index
colors = plt.cm.BuPu(np.linspace(0.1, .9, len(rows)))
n_rows = len(data)

# set width of bar
barWidth = 0.75
fig = plt.subplots(figsize =(12, 8))

# Set position of bar on X axis
br1 = np.arange(len([1]))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

values = [x[0] for x in wy_ca_sum.values]
plt.bar(rows[0], values[0], bar_width, color='r')
plt.bar(rows[1], values[1], bar_width, color='r')
plt.bar(rows[2], values[2], bar_width, color='g')

plt.legend()
plt.show()

# pie chart
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# colors = ['#c9184a', '#a4133c', '#5bc0be']
#ffc6c4,#f4a3a8,#e38191,#cc607d,#ad466c,#8b3058,#672044
# colors = ['#ffc6c4', '#cc607d', '#8b3058']
#A16928,#bd925a,#d6bd8d,#edeac2,#b5c8b8,#79a7ac,#2887a1
colors =['#bd925a','#edeac2', '#2887a1']

fig1, ax1 = plt.subplots()
pt, tx, autotexts = ax1.pie(values, labels=rows, autopct='%1.1f%%',
                            startangle=90, colors=colors)

for i, a in enumerate(autotexts):
    a.set_text("{:.1f}".format(values[i]))

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#     y_offset = y_offset + data[row]
#     cell_text.append(['%1.2f' % x for x in y_offset])

# colors = colors[::-1]
# cell_text.reverse()

# the_table = plt.table(cellText=cell_text,
#                     rowLabels=rows,
#                     rowColours=colors,
#                     colLabels=columns,
#                     loc='bottom')

# the_table.scale(1,2)

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.5, bottom=0.5)
plt.ylabel(element)
plt.xticks([])
plt.title('Flux by Site\n colored by decade')


# facet of these
elements = ['Ca_flux', 'Mg_flux', 'SiO2_Si_flux', 'K_flux', 'Cl_flux']
rows = len(elements)
cols = 3
columns = ('w2', 'w5', 'w6')

fig, ax = plt.subplots(1, 5, figsize=(8, 8))

for i, element in enumerate(elements):
    decader(hbef_wy_sum, 'water_year')
    sum_data = (hbef_wy_sum.pivot_table(index="decade",
                      columns='site_code',
                      values=element))

    # Get some pastel shades for the colors
    data = sum_data.values[1:]
    columns = ('w2', 'w5', 'w6')
    rows = sum_data.index[1:]
    colors = plt.cm.BuPu(np.linspace(0.1, .9, len(rows)))
    n_rows = len(data)


    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        ax[i].bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.2f' % x for x in y_offset])
        ax[i].set_xticks([])
        element_text = element.split('_')[0] + " ${flux}$"
        ax[i].set_title(element_text)
        ax[i].set_ylim([0, 150])

    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    if i == 0:
        the_table = ax[i].table(
                      cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')
    else:
        ax[i].axes.yaxis.set_visible(False)
        the_table = ax[i].table(
                      cellText=cell_text,
                      rowLabels=None,
                      rowColours=None,
                      colLabels=columns,
                      # colWidths = [0.4 for x in columns],
                      loc='bottom')
    the_table.scale(1, 2)

plt.xticks([])
fig.subplots_adjust(bottom=.25)
plt.ylabel(element)

plt.show()

# time series of annual yield
# facet of timeseries
# this shows a timersies of annual flux sums, with a
# gray line showing the difference between experimental annual flux and reference flux

# annual flux time series,
# hbef_wy_sum = hbef.groupby(['water_year', 'site_code']).sum().reset_index()

# and, annual difference in yield for each element
# calculate difference w2-ref yield, w5-ref yield
hbef_wy_sum['Ca_ref'] = 0
hbef_wy_sum.set_index('water_year', 'site_code')

calcium_ref = {
    'w5': [],
    'w2': []
}

# for element in fluxes:
ca_ref = []
for element in fluxes:
    element_ref_list = []
    for index, row in hbef_wy_sum.iterrows():
        if row['site_code'] == 'w5' or row['site_code'] == 'w2':
            year = row['water_year']
            site = row['site_code']
            val = row[element]
            ref = hbef_wy_sum[hbef_wy_sum['water_year'] == year][hbef_wy_sum['site_code']== 'w6'][element].values[0]
            ca_ref_val = val - ref
            element_ref_list.append(ca_ref_val)
        # ca_ref.append([year, site, ca_ref_val])
        else:
            element_ref_list.append(0)

    ele_col = element+'_ref'
    hbef_wy_sum[ele_col] = element_ref_list

fluxes = ['Ca_flux', 'Na_flux', 'Mg_flux', 'Cl_flux', 'K_flux']
cols = hbef_wy_sum.site_code.value_counts().shape[0]
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
        x = hbef_wy_sum[hbef_wy_sum['site_code']==site]['water_year']
        y = hbef_wy_sum[hbef_wy_sum['site_code']==site][element_yield]
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
            ref = hbef_wy_sum[hbef_wy_sum['site_code']==site][fluxref]
            ax[index][i].axvspan(1965, 1967, alpha=0.2,
                                 color='gray',
                                 label='W2 devegetated')
            ax[index][i].plot(x, ref, linewidth=1, color='gray',
                              linestyle=site_col[site][0],
                              label=f'difference annual yield, {site} and reference',
                              alpha=0.5)
        elif site=='w5':
            ref = hbef_wy_sum[hbef_wy_sum['site_code']==site][fluxref]
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

#               entire record

fig, ax = plt.subplots(rows, cols, figsize=(10, 10))
for index, element_yield in enumerate(fluxes):
    for i, site in enumerate(sites):
        x = hbef_ratio[hbef_ratio['site_code']==site]['datetime']
        y = hbef_ratio[hbef_ratio['site_code']==site][element_yield]
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

        # fluxref = element_yield+'_ref'

        if site == 'w2':
            # ref = hbef_ratio[hbef_ratio['site_code']==site][fluxref]
            ax[index][i].axvspan(1965, 1967, alpha=0.2,
                                 color='gray',
                                 label='W2 devegetated')
            # ax[index][i].plot(x, ref, linewidth=1, color='gray',
            #                   linestyle=site_col[site][0],
            #                   label=f'difference annual yield, {site} and reference',
            #                   alpha=0.5)
        elif site=='w5':
            # ref = hbef_ratio[hbef_ratio['site_code']==site][fluxref]
            ax[index][i].axvspan(1983, 1984, alpha=0.2, color='lightgray', label='W5 clearcut and herbicide')
            # ax[index][i].plot(x, ref, linewidth=1, color='gray',
            #                   linestyle=site_col[site][0],
            #                   label=f'difference annual yield, {site} and reference',
            #                   alpha=0.5)

        # ax[index][i].set_ylim([0, 3000])
        # if index == 0:
        #     ax[index][i].set_ylim([0, 80])
        # elif index == 1:
        #     ax[index][i].set_ylim([0, 20])
        # elif index == 2:
        #     ax[index][i].set_ylim([0, 20])
        # elif index == 3:
        #     ax[index][i].set_ylim([0, 15])
        # elif index == 4:
        #     ax[index][i].set_ylim([0, 40])


        if index == len(fluxes)-1:
            ax[index][i].axes.xaxis.set_visible(True)

fig.suptitle("Element Annual Yields Over Time in Hubbard Brook\n1963-2020, interpolated dataset", x=0.42, y=.98)
colors = ['blue', 'red']
lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='-') for c in colors]
labels = ['W6, biogeochemical reference', 'W2 and W5, experimental']
fig.legend(lines, labels, loc='upper right')
plt.show()

#

##       for each element: Ca, Si, Na, Cl, K
#               by year
#               entire record


### look at all ratios above for *all possible molar ratios*
##       for average annual VWC
##       for total annual flux

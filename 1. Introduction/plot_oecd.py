import argparse
import glob
import itertools
import json
from pathlib import Path
import re
import string

import pandas as pd
import numpy as np
import scipy as sp

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import geojson
import geopandas as gpd
import topojson as tp
import imageio
import mapclassify

from pytz import timezone

SCRIPT_DIR = Path(__file__).parent.absolute()
EPSILON = 1e-10

SEOULTZ = timezone('Asia/Seoul')

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"


def plot10():
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR / 'data'
    # OECD
    data_fname = 'PM10_OECD.csv'

    # PM2.5 dataframe
    pdf = pd.read_csv(data_dir / data_fname)

    fig = plt.figure(figsize=(15*0.5, 6.25))

    grouped_data = pdf.groupby(
                ['Country', 'Year'], sort=False, as_index=False)
    df_PM10 = grouped_data.mean()
    print(df_PM10.head(5))


def plot():
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR / 'data'
    targets = ['PM10', 'PM25']

    # OECD
    data_fname = 'PM25_OECD.csv'
    # https://github.com/topojson/world-atlas
    world_fname = 'world.json'
    #world_fname = 'countries-110m.json'
    # https://github.com/eesur/country-codes-lat-long
    country_codes_fname = 'country-codes-lat-long-alpha3.json'

    with open(data_dir / country_codes_fname, 'r') as f:
        country_codes = json.load(f)

    with open(data_dir / world_fname, 'r') as f:
        world = json.load(f)

    # geo pandas
    gdf = gpd.GeoDataFrame.from_file(data_dir / world_fname)
    
    gdf['alpha3'] = gdf.filename.str.split('.', 1, expand=True)[0]
    # gdf.rename(columns={'name': 'county'}, inplace=True)

    # country dataframe
    cdf = pd.DataFrame.from_dict(country_codes['ref_country_codes'])

    # PM2.5 dataframe
    pdf = pd.read_csv(data_dir / data_fname)
    
    df_pm25_line = pdf.copy()
    # # place marker
    # df_pm25_line['markers'] = ['.'] * df_pm25_line.shape[0]
    # df_pm25_line['LOCATION' == 'KOR', 'markers'] = '^'
    # df_pm25_line['LOCATION' == 'OECD', 'markers'] = 'v'

    # Line plot
    fig = plt.figure(figsize=(15*0.5, 6.25))

    grouped_data = df_pm25_line.groupby(
                'LOCATION', sort=False, as_index=False
            )
    labels = list(grouped_data.aggregate(np.sum)['LOCATION'])
    ax = sns.lineplot(x='TIME', y='Value', hue='LOCATION', data=df_pm25_line, legend=False)

    # plt.title(r'$\mathrm{\mathsf{PM}}_{\mathrm{\mathsf{2.5}}}$')
    ax.set_xlabel('YEAR', fontsize='small')
    ax.set_ylabel(
        r'$\mathrm{\mathsf{PM}}_{\mathrm{\mathsf{2.5}}}\;\; \mathrm{(\mu g m^{-3})}$', fontsize='small')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([2000, 2005, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]))
    # add 2019
    x_ticks = np.append(ax.get_xticks(), 2019)
    ax.set_xticks(x_ticks)
    # add label to axes manually
    for l, line in zip(labels, ax.lines[:]):
        line.set_label(l)

    # add hline for WHO guideline
    wholines = ax.hlines(10.0, xmin=2000, xmax=2019, # transform=ax.get_xaxis_transform(),
                label='WHO', colors=['#1A4E6665'], linestyles=['--'], linewidths=[2.0])
    
    for i,line in enumerate(ax.lines[:]):
        if line.get_label() == 'KOR':
            ax.lines[i].set_color('tab:orange')
            ax.lines[i].set_marker('^')
            ax.lines[i].set_markersize(7)
            ax.lines[i].set_linewidth(3)
        elif line.get_label() == 'OECD':
            ax.lines[i].set_color('tab:blue')
            ax.lines[i].set_marker('v')
            ax.lines[i].set_markersize(7)
            ax.lines[i].set_linewidth(3)
        elif line.get_label() == 'WHO':
            ax.lines[i].set_color('#1A4E66')
            ax.lines[i].set_linestyle('dashed')
            ax.lines[i].set_linewidth(2)
        else:
            ax.lines[i].set_color('#1A4E66')
            ax.lines[i].set_linewidth(1)
            ax.lines[i].set_alpha(0.3)

    _leg_handles, _leg_labels = ax.get_legend_handles_labels()
    leg_handles, leg_labels = [], []
    for lh, ll in zip(_leg_handles, _leg_labels):
        if ll == 'KOR':
            leg_handles.append(lh)
            leg_labels.append('Korea')
        if ll == 'OECD':
            leg_handles.append(lh)
            leg_labels.append('OECD average')
    leg_handles.append(wholines)
    leg_labels.append('WHO threshold')
    ax.legend(leg_handles, leg_labels,
              loc='upper left',
              fancybox=True,
              fontsize='medium')

    filename = 'PM25_lineplot'
    plt.savefig(output_dir / (filename + '.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.savefig(output_dir / (filename + '.svg'), bbox_inches='tight', pad_inches=0.1)
    plt.close()


    pdf['alpha3'] = pdf['LOCATION']
    df1 = pdf.merge(cdf, on='alpha3')

    df1 = df1.drop(columns=['INDICATOR','SUBJECT','MEASURE','FREQUENCY','Flag Codes'])
    df1 = df1.drop(columns=['alpha2', 'country', 'latitude', 'longitude'])
    df1 = df1.rename(columns={'numeric': 'id'})

    # remove small countries by filtering None
    #gdf = gdf[gdf.id.apply(lambda x: x != None)]
    #gdf = gdf.astype({'id': 'int64'})
    # df1 = df1.astype({'id': 'object'})

    df = gdf.merge(df1, how='outer', on='alpha3')
    df['Value'] = df['Value'].fillna(-1)
    df_nan = df.loc[(df['TIME'].isnull())]
    times = [2000,2005] + list(range(2010,2020))
    for t in times:
        print(t)
        df_time = df.loc[(df['TIME'] == t)]
        scheme = mapclassify.EqualInterval(df_time['Value'], k=5)
        df_time = pd.concat([df_time, df_nan])

        fig = plt.figure(figsize=(7.22, 9.45))
        ax = plt.gca()
        # Projection Error
        # https://stackoverflow.com/a/60126085
        gplt.choropleth(
            df_time,
            hue='Value',
            scheme=scheme,
            cmap='Blues',
            linewidth=0.1,
            edgecolor='black',
            projection=gcrs.Miller(),
            legend=True
        )
        leg_handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(leg_handles, leg_labels, fontsize='x-small')
        #projection=gcrs.Miller()
        plt.title(r'$\mathrm{\mathsf{PM}}_{\mathrm{\mathsf{2.5}}}$, ' + str(t))
        filename = 'PM25_Choropleth_'+ str(t)
        plt.savefig(output_dir / (filename + '.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)
        plt.savefig(output_dir / (filename + '.svg'), bbox_inches='tight', pad_inches=0.1)
        plt.close()


if __name__ == '__main__':
    plot()
    # plot10()



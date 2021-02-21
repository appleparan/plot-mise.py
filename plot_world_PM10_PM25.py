import argparse
import glob
import itertools
from pathlib import Path
import re
import string

import pandas as pd
import numpy as np
import scipy as sp

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.metrics
import sklearn

import statsmodels.api as sm
import statsmodels.tsa.stattools as tsast
import statsmodels.graphics.tsaplots as tpl

import bokeh
from bokeh.models import Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png, export_svgs
from bokeh.palettes import Category10, Category20

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import mpl_toolkits.axes_grid1.parasite_axes as paxes

SCRIPT_DIR = Path(__file__).parent.absolute()
EPSILON = 1e-10

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"


def plot_(df, df_grp, output_dir, target='PM10'):
    def get_color_map():
        # create color from FFFFFF to 1A4E66
        color_vals_b = np.ones((256, 4))
        color_vals_b[:, 0] = np.linspace(26/256, 1, 256)
        color_vals_b[:, 1] = np.linspace(78/256, 1, 256)
        color_vals_b[:, 2] = np.linspace(102/256, 1, 256)
        color_vals_g = np.ones((256, 4))
        color_vals_g[:, 0] = np.linspace(50/256, 1, 256)
        color_vals_g[:, 1] = np.linspace(69/256, 1, 256)
        color_vals_g[:, 2] = np.linspace(77/256, 1, 256)
        # need to reversed!
        # small values -> white
        # large values -> dark blue
        # deep blue
        cm_db = mcolors.ListedColormap(color_vals_b).reversed()
        cm_dg = mcolors.ListedColormap(color_vals_g).reversed()

        return cm_db, cm_dg

    # sort first by Region and values
    df.sort_values(by=['Region', target], inplace=True)

    cm_db, cm_dg = get_color_map()
    fig = plt.figure(figsize=(15.4, 7.2), dpi=600)
    ax = fig.gca()
    ax.set_xlim((-1, 39))
    values = sorted(list(df_grp[target]))
    maxval_1 = values[-1]
    maxval_2 = values[-2]
    norm = mcolors.Normalize(0.0, maxval_2)
    max_color = '#E26C2285'
    seoul_color = '#D93839'

    bars = ax.bar(x=df['City'], height=df[target])
    cities_noseoul = df['City'].values.tolist()
    cities_noseoul.remove('Seoul')
    ticks = ax.get_xticks()
    labels = list(df['City'])

    ax.xaxis.set_ticks(ticks)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))    
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(labelsize='large')
    
    # region based visible ticks
    par1 = ax.twiny()
    par1.set_frame_on(True)
    par1.patch.set_visible(False)
    par1.set_xlim((ax.get_xlim()))
    par1.xaxis.set_ticks_position("bottom")
    par1.xaxis.set_label_position("bottom")
    for sp in par1.spines.values():
        sp.set_visible(False)
    par1.spines["bottom"].set_position(("axes", -0.21))
    par1.spines["top"].set_visible(True)
    par1.xaxis.set_major_locator(mticker.FixedLocator([2.5, 6.5, 10.5, 14.5, 18.5,
                                                        22.5, 26.5, 30.5, 34.5]))
    par1.xaxis.set_ticklabels([])                                                        
    par1.xaxis.set_tick_params(direction='out', length=12, width=1.5, grid_alpha=0.0)
    
    # region based invisible ticks
    par2 = ax.twiny()
    par2.set_frame_on(True)
    par2.patch.set_visible(False)
    par2.set_xlim((ax.get_xlim()))
    par2.xaxis.set_ticks_position("bottom")
    par2.xaxis.set_label_position("bottom")
    for sp in par2.spines.values():
        sp.set_visible(False)
    par2.spines["bottom"].set_position(("axes", -0.21))
    # par2.new_fixed_axis(loc="bottom", offset=offset)
    par2.spines["top"].set_visible(False)
    par2.spines["bottom"].set_visible(False)
    
    # Region ticks
    par2.xaxis.set_major_locator(mticker.FixedLocator([1, 4.5, 8.5, 12.5, 16.5,
                                                       20.5, 24.5, 28.5, 32.5, 36.5]))
    par2.xaxis.set_major_formatter(mticker.FixedFormatter(list(df_grp.index)))
    par2.xaxis.set_tick_params(grid_alpha=0.0, length=0.0, labelsize='large')

    ccv = mcolors.ColorConverter()
    for bar, label in zip(bars, ax.xaxis.get_ticklabels(which='both')):
        # label is a Text instance
        label.set_rotation(90)
        region = df.loc[df['City'] == label.get_text(), 'Region'].values
        region_idx = list(df_grp.index).index(region)
        region_value = df_grp.loc[region, target].values[0]

        if label.get_text() in cities_noseoul:    
            if region_idx % 2 == 0:
                c = cm_db(norm(region_value), alpha=1)
            else:
                c = cm_dg(norm(region_value), alpha=1)
            # print(c, region_value, maxval, region_value / maxval, label.get_text())
            bar.set_color(c)
            # label.set_color(c)

        if region_value == maxval_1:
            bar.set_color(max_color)
            # label.set_color(max_color)

        if label.get_text() == 'Seoul':
            bar.set_color(seoul_color)
            label.set_color(seoul_color)
       
    fig.tight_layout()
    sns.despine()
    output_to_plot = "world_" + target
    png_path = output_dir / (output_to_plot + '.png')
    svg_path = output_dir / (output_to_plot + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()

def plot():
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR / 'data'

    filename = 'PM10_PM25_region_2014.csv'
    df = pd.read_csv(data_dir / filename)
    print(df.head(5))

    df_grp = df.groupby('Region').mean()
    print(df_grp)
    plot_(df, df_grp, output_dir=output_dir, target='PM10')
    plot_(df, df_grp, output_dir=output_dir, target='PM25')

if __name__ == '__main__':
    plot()

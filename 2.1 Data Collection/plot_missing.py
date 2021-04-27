import argparse
import datetime as dt
import glob
import itertools
from pathlib import Path
import re
import string

import pytz

import pandas as pd
import numpy as np
import scipy as sp

import sklearn

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from matplotlib.patches import Circle, Ellipse, Rectangle

SCRIPT_DIR = Path(__file__).parent.absolute()
EPSILON = 1e-10

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"

TARGET_MAP = {
    "SO2": r'\mathrm{\mathsf{SO}}_{2}',
    "CO": r'\mathrm{\mathsf{CO}}',
    "O3": r'\mathrm{\mathsf{O}}}_{3}',
    "NO2": r'\mathrm{\mathsf{NO}}_{2}',
    'PM10': r'\mathrm{\mathsf{PM}}_{10}',
    'PM25': r'\mathrm{\mathsf{PM}}_{2.5}',
    'temp': r'\mathrm{\mathsf{Temperature}}',
    'u': r'\mathrm{\mathsf{Wind\ Speed\ (Zonal)}}',
    'v': r'\mathrm{\mathsf{Wind\ Speed\ (Meridional)}}',
    'wind_spd': '\mathrm{\mathsf{Wind\ Speed}}',
    'wind_dir': '\mathrm{\mathsf{Wind\ Direction}}',
    'wind_sdir': '\mathrm{\mathsf{Wind\ Direction(sin)}}',
    'wind_cdir': '\mathrm{\mathsf{Wind\ Direction(cos)}}',
    'pres': r'\mathrm{\mathsf{Pressure}}',
    'humid': r'\mathrm{\mathsf{Relative\ Humidity}}',
    'prep': r'\mathrm{\mathsf{Precipitation}}'
}

def plot(station_name='종로구'):
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR / 'data'
    features = ['SO2', 'CO', 'NO2', 'PM10', 'PM25', 'temp', 'wind_spd', 'wind_dir', 'pres', 'humid', 'prep']

    df = pd.read_csv(data_dir / f'df_raw_no_impute.csv', index_col=0)
    df = df.loc[:, features]

    #
    total_length = len(df.index)
    df_isna = df.isna().sum()
    df_isnotna = df.notna().sum()

    pct_missing = {}
    for fea in features:
        pct_missing[fea] = df_isna[fea] / total_length * 100

    # https://stackoverflow.com/a/37543737
    for fea in features:
        df.loc[df.loc[:, fea].notna(), fea] = 1

    nrows = 1
    ncols = 1

    # rough figure size
    w_pad, h_pad = 0.1, 0.30
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    # ax_size = min(7.22 / ncols, 9.45 / nrows)
    ax_size = min(7.22 / ncols, 7.22 / nrows)
    # legend_size = 0.6 * fig_size
    fig_size_w = ax_size*ncols
    fig_size_h = ax_size*nrows
    width = 0.5

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(7.22, 5.415),
                            dpi=600,
                            frameon=False,
                            subplot_kw={
                                'clip_on': False
                            })

    fig.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.2)

    sns.heatmap(df.isnull(), cmap=['tab:blue', 'white'], cbar=False, ax=axs)
    ymin = axs.get_ylim()[0]
    for i, fea in enumerate(features):
        axs.annotate('{0: .1f}%'.format(pct_missing[fea]),
                    color='black',
                    xy=(i + 0.5, ymin),
                    textcoords='offset points',
                    xytext=(0, 11),
                    # xytext=(0.5, -0.5),  # use 3 points offset
                    # fontweight='bold',
                    # textcoords="offset points",  # in both directions
                    ha="center", va="top")

    labels = [rf'${TARGET_MAP[tick.get_text()]}$' for tick in axs.get_xticklabels()]
    axs.set_xticklabels(labels, rotation=70)
    axs.invert_yaxis()
    axs.set_yticks([])
    axs.set_ylabel('')

    output_prefix = f'{station_name}_missing'
    png_path = output_dir / (output_prefix + '.png')
    # svg_path = output_dir / (output_prefix + '.svg')
    plt.savefig(png_path, dpi=600)
    # plt.savefig(svg_path)
    plt.close(fig)

if __name__ == '__main__':
    plot(station_name='종로구')
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
    'wind_sdir': '\mathrm{\mathsf{Wind\ Direction(sin)}}',
    'wind_cdir': '\mathrm{\mathsf{Wind\ Direction(cos)}}',
    'pres': r'\mathrm{\mathsf{Pressure}}',
    'humid': r'\mathrm{\mathsf{Relative\ Humidity}}',
    'prep': r'\mathrm{\mathsf{Rainfall}}',
    'snow': r'\mathrm{\mathsf{Snow}}'
}

def plot(station_name='종로구'):
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR / 'data'

    stat_dir = data_dir / 'impute_stats'

    df_na = pd.read_csv(stat_dir / f'stats_{station_name}_isna.csv', index_col=0)
    df_notna = pd.read_csv(stat_dir / f'stats_{station_name}_isnotna.csv', index_col=0)

    df_na.rename(columns={'0': 'na'}, inplace=True)
    df_notna.rename(columns={'0': 'notna'}, inplace=True)

    df = df_na.merge(df_notna, left_index=True, right_index=True)
    df['total'] = df['na'] + df['notna']
    df['na_pct'] = df['na'] / (df['na'] + df['notna']) * 100
    df['notna_pct'] = df['notna'] / (df['na'] + df['notna']) * 100
    features = ['SO2', 'CO', 'NO2', 'PM10', 'PM25', 'temp', 'wind_spd', 'wind_sdir', 'wind_cdir', 'pres', 'humid', 'prep']

    df = df.loc[features, :]

    nrows = 1
    ncols = 1
    multipanel_labels = np.array(list(string.ascii_uppercase)[:(nrows*ncols)]).reshape(nrows, ncols)
    multipanellabel_position = (-0.08, 1.02)

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
    fig.subplots_adjust(left=0.1, bottom=0.15)

    bar1 = axs.bar(list(range(len(features))), df['na_pct'], color='tab:gray', tick_label=features, alpha=0.3, label='Missing')
    bar2 = axs.bar(list(range(len(features))), df['notna_pct'], bottom=df['na_pct'], color='tab:blue', tick_label=features, label='Not Missing')

    for rect1, rect2 in zip(bar1, bar2):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        axs.annotate('{0: .1f}%'.format(height1),
                        color='white',
                        xy=(rect1.get_x() + rect1.get_width() / 2, (height1 + 0.5)),
                        xytext=(-1.5, 0),  # use 3 points offset
                        fontweight='bold',
                        textcoords="offset points",  # in both directions
                        ha='center', va='bottom')

    labels = [rf'${TARGET_MAP[tick.get_text()]}$' for tick in axs.get_xticklabels()]
    axs.set_xticklabels(labels, rotation=70)
    # axs.margins(0.3)    
    axs.set_ylabel('Percentage (%)')
    axs.legend()

    output_prefix = f'{station_name}_missingbar'
    png_path = output_dir / (output_prefix + '.png')
    svg_path = output_dir / (output_prefix + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close(fig)

if __name__ == '__main__':
    plot(station_name='종로구')
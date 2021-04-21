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
    "SO2": '\mathrm{\mathsf{SO}}_{2}',
    "CO": '\mathrm{\mathsf{CO}}',
    "O3": '\mathrm{\mathsf{O}}}_{3}',
    "NO2": '\mathrm{\mathsf{NO}}_{2}',
    'PM10': '\mathrm{\mathsf{PM}}_{10}',
    'PM25': '\mathrm{\mathsf{PM}}_{2.5}',
    'temp': '\mathrm{\mathsf{Temperature}}',
    'wind_spd': '\mathrm{\mathsf{Wind\ Speed}}',
    'wind_sdir': '\mathrm{\mathsf{Wind\ Speed(sin)}}',
    'wind_cdir': '\mathrm{\mathsf{Wind\ Speed(cos)}}',
    'pres': '\mathrm{\mathsf{Pressure}}',
    'humid': '\mathrm{\mathsf{Rel.\ Humidity}}',
    'prep': '\mathrm{\mathsf{Precipitation}}',
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
    print(df.head(18))

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
                            figsize=(ax_size*ncols, ax_size*nrows),
                            dpi=600,
                            frameon=False,
                            subplot_kw={
                                'clip_on': False,
                                'box_aspect': 1
                            })

    fig.tight_layout(w_pad=w_pad, h_pad=h_pad)
    fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

    bar1 = axs.bar(list(range(len(features))), df['notna_pct'], color='tab:blue', alpha=0.3, tick_label=features, label='Not Missing')
    bar2 = axs.bar(list(range(len(features))), df['na_pct'], bottom=df['notna_pct'], color='tab:orange', tick_label=features, label='Missing')

    for rect1, rect2 in zip(bar1, bar2):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        if height2 != 0:
            axs.annotate('{0: .1f}%'.format(height2),
                         color='black',
                         xy=(rect1.get_x() + rect1.get_width() / 2, (height1 + height2) / 2),
                         xytext=(-1.5, 0),  # use 3 points offset
                         fontweight='bold',
                         textcoords="offset points",  # in both directions
                         ha='center', va='bottom')

    labels = [rf'${TARGET_MAP[tick.get_text()]}$' for tick in axs.get_xticklabels()]
    axs.set_xticklabels(labels, rotation=45)
    # axs.margins(0.3)    
    axs.set_ylabel('Percentage (%)')
    axs.legend()

    plt.subplots_adjust(bottom=0.15)
    fig.tight_layout()
    output_prefix = f'{station_name}_missingbar'
    png_path = output_dir / (output_prefix + '.png')
    svg_path = output_dir / (output_prefix + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close(fig)

if __name__ == '__main__':
    plot(station_name='종로구')
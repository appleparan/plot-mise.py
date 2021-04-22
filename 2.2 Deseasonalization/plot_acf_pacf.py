import argparse
import datetime as dt
import glob
import itertools
from pathlib import Path
import re
import string

from pytz import timezone

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
import matplotlib.ticker as mticker
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsast
import statsmodels.graphics.tsaplots as tpl

import data

SCRIPT_DIR = Path(__file__).parent.absolute()

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"

INPUTDATA_path = SCRIPT_DIR / '..' / 'data'
SEOULTZ = timezone('Asia/Seoul')

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

def plot():
    jongno_fname = 'input_jongno_imputed_hourly_pandas.csv'
    seoul_fname = 'input_seoul_imputed_hourly_pandas.csv'

    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    plot_data_dir = SCRIPT_DIR / 'out' / 'csv'
    Path.mkdir(plot_data_dir, parents=True, exist_ok=True)
    plot_png_dir = SCRIPT_DIR / 'out' / 'png'
    Path.mkdir(plot_png_dir, parents=True, exist_ok=True)
    plot_svg_dir = SCRIPT_DIR / 'out' / 'svg'
    Path.mkdir(plot_svg_dir, parents=True, exist_ok=True)

    stations = ['종로구']
    targets = ['PM10', 'PM25']
    nlags = 7

    # 1 2 3 4
    # 5 6 7 8
    # PM10: 1, 2, 3
    # PM25: 4, 5, 6
    # 1, 5: ACF (raw)
    # 2, 6: PACF (raw)
    # 3, 7: ACF (desea)
    # 4, 8: PACF (desea)
    # Seasonality
    nrows = 2
    ncols = 4
    if nrows == 1 or ncols == 1:
        # 1D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:ncols])
    else:
        # 2D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:(nrows*ncols)]).reshape(nrows, ncols)
    multipanellabel_position = (-0.08, 1.02)

    # rough figure size
    w_pad, h_pad = 1.08, 1.08
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    # ax_size = min(7.22 / ncols, 9.45 / nrows)
    ax_size = min(14.44 / ncols, 7.22 / nrows)
    # legend_size = 0.6 * fig_size
    fig_size_w = ax_size*ncols
    fig_size_h = ax_size*nrows

    train_valid_fdate = dt.datetime(2008, 1, 3, 1).astimezone(SEOULTZ)
    train_valid_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)

    for station_name in stations:
        fig, axs = plt.subplots(nrows, ncols,
            figsize=(ax_size*ncols, ax_size*nrows),
            dpi=600,
            frameon=False,
            subplot_kw={
                'clip_on': False,
                'box_aspect': 1
            })

        # keep right distance between subplots
        fig.tight_layout(w_pad=w_pad, h_pad=h_pad)
        fig.subplots_adjust(left=0.1, bottom=0.15, top=0.9)

        for rowi, target in enumerate(targets):
            Path.mkdir(plot_data_dir / target, parents=True, exist_ok=True)
            Path.mkdir(plot_png_dir  / target, parents=True, exist_ok=True)
            Path.mkdir(plot_svg_dir  / target, parents=True, exist_ok=True)
            dataset = data.UnivariateRNNMeanSeasonalityDataset(
                            station_name=station_name,
                            target=target,
                            filepath=INPUTDATA_path / seoul_fname,
                            features=[target],
                            fdate=train_valid_fdate,
                            tdate=train_valid_tdate,
                            sample_size=48,
                            output_size=24)
            dataset.preprocess()

            df_raw = dataset.ys_raw
            df_res = dataset.ys

            tpl.plot_acf(df_raw[target], ax=axs[rowi, 0], fft=True, lags=nlags*24,
                         use_vlines=False, marker=None, linestyle='solid', linewidth=1)
            tpl.plot_acf(df_res[target], ax=axs[rowi, 2], fft=True, lags=nlags*24,
                         use_vlines=False, marker=None, linestyle='solid', linewidth=1)

            axs[rowi, 0].set_ylabel(r'$C(s)$ - $\mathrm{{{0:s}}}$'.format(TARGET_MAP[target]), fontsize='large')

            tpl.plot_pacf(df_raw[target], ax=axs[rowi, 1], lags=12,
                          use_vlines=True, markersize=2)
            tpl.plot_pacf(df_res[target], ax=axs[rowi, 3], lags=12,
                          use_vlines=True, markersize=2)

            if rowi == nrows-1:
                for coli in range(ncols):
                    axs[rowi, coli].set_xlabel(r'lag $s$ (hour)', fontsize='large')

            # ACF
            axs[rowi, 0].xaxis.set_major_locator(mticker.MultipleLocator(24))
            axs[rowi, 2].xaxis.set_major_locator(mticker.MultipleLocator(24))
            # PACF
            axs[rowi, 1].xaxis.set_major_locator(mticker.MultipleLocator(6))
            axs[rowi, 3].xaxis.set_major_locator(mticker.MultipleLocator(6))
            for coli in range(ncols):
                axs[rowi, coli].set_title("")
                axs[rowi, coli].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
                axs[rowi, coli].annotate(multipanel_labels[rowi, coli], (-0.08, 1.02), xycoords='axes fraction',
                                fontsize='large', fontweight='bold')

                for tick in axs[rowi, coli].xaxis.get_major_ticks():
                    tick.label.set_fontsize('medium')
                for tick in axs[rowi, coli].yaxis.get_major_ticks():
                    tick.label.set_fontsize('medium')

            # dataset.plot_seasonality(plot_data_dir / target, plot_png_dir / target, plot_svg_dir / target)
        fig.tight_layout()
        output_fname = f"{station_name}_acf_pacf"
        png_path = output_dir / (output_fname + '.png')
        svg_path = output_dir / (output_fname + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

if __name__ == '__main__':
    plot()

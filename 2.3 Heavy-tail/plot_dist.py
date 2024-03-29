import argparse
import datetime as dt
from decimal import Decimal
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

# heavy tailed distribution power law
import powerlaw

import data

SCRIPT_DIR = Path(__file__).parent.absolute()

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

INPUTDATA_path = SCRIPT_DIR / '..' / 'data'
SEOULTZ = timezone('Asia/Seoul')

def plot_ccdf(targets=['PM10', 'PM25'], sample_size=48, output_size=24):
    assert len(targets) > 1
    jongno_fname = 'input_jongno_imputed_hourly_pandas.csv'
    seoul_fname = 'input_seoul_imputed_hourly_pandas.csv'

    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    stations = ['종로구']

    # plot distribution of target (raw data)
    nrows = 2
    ncols = 2
    if nrows == 1 or ncols == 1:
        # 1D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:ncols])
    else:
        # 2D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:(nrows*ncols)]).reshape(nrows, ncols)
    multipanellabel_position = (-0.08, 1.02)

    # rough figure size
    w_pad, h_pad = 0.1, 0.30
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    # ax_size = min(7.22 / ncols, 9.45 / nrows)
    ax_size = min(7.22 / ncols, 9.45 / nrows)
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
        fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

        for rowi, target in enumerate(targets):
            dataset = data.UnivariateRNNMeanSeasonalityDataset(
                            station_name=station_name,
                            target=target,
                            filepath=INPUTDATA_path / seoul_fname,
                            features=[target],
                            fdate=train_valid_fdate,
                            tdate=train_valid_tdate,
                            sample_size=sample_size,
                            output_size=output_size)
            dataset.preprocess()

            df_raw = dataset.ys_raw
            df_res = dataset.ys

            arr_raw = df_raw[target].to_numpy()
            arr_res = df_res[target].to_numpy()
            dist_raw = np.random.normal(np.mean(arr_raw), np.std(arr_raw), len(arr_raw))
            dist_res = np.random.normal(np.mean(arr_res), np.std(arr_res), len(arr_res))
            df_plot_raw = pd.DataFrame.from_dict({target: arr_raw, 'gaussian': dist_raw})
            df_plot_res = pd.DataFrame.from_dict({target: arr_res, 'gaussian': dist_res})

            raw_fit = powerlaw.Fit(data=df_plot_raw[target].to_numpy(), discrete=True, xmin=1, xmax=None)
            res_fit = powerlaw.Fit(data=df_plot_res[target].to_numpy(), discrete=True, xmin=1, xmax=None)

            raw_fit.plot_ccdf(ax=axs[rowi, 0], color='tab:blue', lw=2, label='Empirical Data')
            raw_fit.power_law.plot_ccdf(ax=axs[rowi, 0], color='tab:orange', lw=2, ls='solid', label='Power-Law Fit')
            raw_fit.lognormal.plot_ccdf(ax=axs[rowi, 0], color='tab:orange', lw=2, ls='--', label='Log-Normal Fit')
            print(f"{target} - raw power law - xmin, alpha : {raw_fit.power_law.xmin}, {raw_fit.power_law.alpha}")
            print(f"{target} - raw lognormal - mu, sigma : {raw_fit.lognormal.mu}, {raw_fit.lognormal.sigma}")
            res_fit.plot_ccdf(ax=axs[rowi, 1], color='tab:blue', lw=2, label='Empirical Data')
            res_fit.power_law.plot_ccdf(ax=axs[rowi, 1], color='tab:orange', lw=2, ls='solid', label='Power-Law Fit')
            res_fit.lognormal.plot_ccdf(ax=axs[rowi, 1], color='tab:orange', lw=2, ls='--', label='Log-Normal Fit')
            print(f"{target} - res power law - xmin, alpha : {res_fit.power_law.xmin}, {res_fit.power_law.alpha}")
            print(f"{target} - res lognormal - mu, sigma : {res_fit.lognormal.mu}, {res_fit.lognormal.sigma}")

            axs[0, 0].set_title('Raw')
            axs[0, 1].set_title('Deseasonalized')

            # disable y label on right side plot
            axs[rowi, 0].set_ylabel(f"CCDF", fontsize='medium')
            axs[rowi, 1].yaxis.label.set_visible(False)

            # remove legend title
            for coli in range(2):
                # axs[rowi, coli].legend(loc='best')
                axs[rowi, coli].set_xlabel(rf"${TARGET_MAP[target]}$", fontsize='medium')
                axs[rowi, coli].legend()
                axs[rowi, coli].grid(True, zorder=-5)
                # axs[rowi, coli].set_xlabel(target)
                axs[rowi, coli].set_yscale('log')
                axs[rowi, coli].annotate(multipanel_labels[rowi, coli], (-0.08, 1.05), xycoords='axes fraction',
                                fontsize='medium', fontweight='bold')
                # small yticks
                for tick in axs[rowi, coli].yaxis.get_major_ticks():
                    tick.label.set_fontsize('small')
                for tick in axs[rowi, coli].xaxis.get_major_ticks():
                    tick.label.set_fontsize('small')

        fig.tight_layout()
        output_prefix = f'{station_name}_ccdf'
        png_path = output_dir / (output_prefix + '.png')
        svg_path = output_dir / (output_prefix + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close(fig)

def plot_cdf(targets=['PM10', 'PM25'], sample_size=48, output_size=24):
    assert len(targets) > 1
    jongno_fname = 'input_jongno_imputed_hourly_pandas.csv'
    seoul_fname = 'input_seoul_imputed_hourly_pandas.csv'

    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    stations = ['종로구']

    # plot distribution of target (raw data)
    nrows = 2
    ncols = 2
    if nrows == 1 or ncols == 1:
        # 1D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:ncols])
    else:
        # 2D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:(nrows*ncols)]).reshape(nrows, ncols)
    multipanellabel_position = (-0.08, 1.02)

    # rough figure size
    w_pad, h_pad = 0.1, 0.30
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    # ax_size = min(7.22 / ncols, 9.45 / nrows)
    ax_size = min(7.22 / ncols, 9.45 / nrows)
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
        fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

        for rowi, target in enumerate(targets):
            dataset = data.UnivariateRNNMeanSeasonalityDataset(
                            station_name=station_name,
                            target=target,
                            filepath=INPUTDATA_path / seoul_fname,
                            features=[target],
                            fdate=train_valid_fdate,
                            tdate=train_valid_tdate,
                            sample_size=sample_size,
                            output_size=output_size)
            dataset.preprocess()

            df_raw = dataset.ys_raw
            df_res = dataset.ys

            arr_raw = df_raw[target].to_numpy()
            arr_res = df_res[target].to_numpy()
            dist_raw = np.random.normal(np.mean(arr_raw), np.std(arr_raw), len(arr_raw))
            dist_res = np.random.normal(np.mean(arr_res), np.std(arr_res), len(arr_res))
            df_plot_raw = pd.DataFrame.from_dict({target: arr_raw, 'gaussian': dist_raw})
            df_plot_res = pd.DataFrame.from_dict({target: arr_res, 'gaussian': dist_res})
            raw_fit = powerlaw.Fit(data=df_plot_raw[target].to_numpy(), discrete=True, xmin=1, xmax=None)
            res_fit = powerlaw.Fit(data=df_plot_res[target].to_numpy(), discrete=True, xmin=1, xmax=None)

            raw_fit.plot_cdf(ax=axs[rowi, 0], color='tab:blue', lw=2, label='Empirical Data')
            raw_fit.power_law.plot_cdf(ax=axs[rowi, 0], color='tab:orange', lw=2, ls='solid', label='Power-Law Fit')
            raw_fit.lognormal.plot_cdf(ax=axs[rowi, 0], color='tab:orange', lw=2, ls='--', label='Log-Normal Fit')
            print(f"{target} - raw power law - xmin, alpha : {raw_fit.power_law.xmin}, {raw_fit.power_law.alpha}")
            print(f"{target} - raw lognormal - mu, sigma : {raw_fit.lognormal.mu}, {raw_fit.lognormal.sigma}")
            res_fit.plot_cdf(ax=axs[rowi, 1], color='tab:blue', lw=2, label='Empirical Data')
            res_fit.power_law.plot_cdf(ax=axs[rowi, 1], color='tab:orange', lw=2, ls='solid', label='Power-Law Fit')
            res_fit.lognormal.plot_cdf(ax=axs[rowi, 1], color='tab:orange', lw=2, ls='--', label='Log-Normal Fit')
            print(f"{target} - res power law - xmin, alpha : {res_fit.power_law.xmin}, {res_fit.power_law.alpha}")
            print(f"{target} - res lognormal - mu, sigma : {res_fit.lognormal.mu}, {res_fit.lognormal.sigma}")

            axs[0, 0].set_title('Raw')
            axs[0, 1].set_title('Deseasonlized')

            # disable y label on right side plot
            axs[rowi, 0].set_ylabel(f"CDF - " + rf"${TARGET_MAP[target]}$")
            axs[rowi, 1].yaxis.label.set_visible(False)

            # remove legend title
            for coli in range(2):
                # axs[rowi, coli].legend(loc='best')
                axs[rowi, coli].legend()
                axs[rowi, coli].grid(True, zorder=-5)
                axs[rowi, coli].set_xlabel('')
                axs[rowi, coli].set_xscale('linear')
                axs[rowi, coli].set_yscale('linear')
                axs[rowi, coli].annotate(multipanel_labels[rowi, coli], (-0.08, 1.05), xycoords='axes fraction',
                                fontsize='medium', fontweight='bold')

                # small yticks
                for tick in axs[rowi, coli].yaxis.get_major_ticks():
                    tick.label.set_fontsize('small')
                for tick in axs[rowi, coli].xaxis.get_major_ticks():
                    tick.label.set_fontsize('small')

        fig.tight_layout()
        output_prefix = f'{station_name}_cdf'
        png_path = output_dir / (output_prefix + '.png')
        svg_path = output_dir / (output_prefix + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close(fig)

def plot_fit_pdf(targets=['PM10', 'PM25'], sample_size=48, output_size=24):
    assert len(targets) > 1
    jongno_fname = 'input_jongno_imputed_hourly_pandas.csv'
    seoul_fname = 'input_seoul_imputed_hourly_pandas.csv'

    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    stations = ['종로구']

    # plot distribution of target (raw data)
    nrows = 2
    ncols = 2
    if nrows == 1 or ncols == 1:
        # 1D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:ncols])
    else:
        # 2D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:(nrows*ncols)]).reshape(nrows, ncols)
    multipanellabel_position = (-0.08, 1.02)

    # rough figure size
    w_pad, h_pad = 0.2, 0.30
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    # ax_size = min(7.22 / ncols, 9.45 / nrows)
    ax_size = min(7.22 / ncols, 9.45 / nrows)
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
        fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

        for rowi, target in enumerate(targets):
            dataset = data.UnivariateRNNMeanSeasonalityDataset(
                            station_name=station_name,
                            target=target,
                            filepath=INPUTDATA_path / seoul_fname,
                            features=[target],
                            fdate=train_valid_fdate,
                            tdate=train_valid_tdate,
                            sample_size=sample_size,
                            output_size=output_size)
            dataset.preprocess()

            df_raw = dataset.ys_raw
            df_res = dataset.ys

            arr_raw = df_raw[target].to_numpy()
            arr_res = df_res[target].to_numpy()
            dist_raw = np.random.normal(np.mean(arr_raw), np.std(arr_raw), len(arr_raw))
            dist_res = np.random.normal(np.mean(arr_res), np.std(arr_res), len(arr_res))
            df_plot_raw = pd.DataFrame.from_dict({target: arr_raw, 'gaussian': dist_raw})
            df_plot_res = pd.DataFrame.from_dict({target: arr_res, 'gaussian': dist_res})

            raw_fit = powerlaw.Fit(data=df_plot_raw[target].to_numpy(), discrete=True, xmin=1, xmax=None)
            res_fit = powerlaw.Fit(data=df_plot_res[target].to_numpy(), discrete=True, xmin=1, xmax=None)

            raw_fit.plot_pdf(ax=axs[rowi, 0], color='tab:blue', lw=2, label='Empirical Data')
            raw_fit.power_law.plot_pdf(ax=axs[rowi, 0], color='tab:green', lw=1.5, ls='solid', label='Power-Law Fit')
            raw_fit.lognormal.plot_pdf(ax=axs[rowi, 0], color='tab:orange', lw=1.5, ls='--', label='Log-Normal Fit')
            print(f"{target} - raw power law - xmin, alpha : {raw_fit.power_law.xmin}, {raw_fit.power_law.alpha}")
            print(f"{target} - raw lognormal - mu, sigma : {raw_fit.lognormal.mu}, {raw_fit.lognormal.sigma}")
            res_fit.plot_pdf(ax=axs[rowi, 1], color='tab:blue', lw=2, label='Empirical Data')
            res_fit.power_law.plot_pdf(ax=axs[rowi, 1], color='tab:green', lw=1.5, ls='solid', label='Power-Law Fit')
            res_fit.lognormal.plot_pdf(ax=axs[rowi, 1], color='tab:orange', lw=1.5, ls='--', label='Log-Normal Fit')
            print(f"{target} - res power law - xmin, alpha : {res_fit.power_law.xmin}, {res_fit.power_law.alpha}")
            print(f"{target} - res lognormal - mu, sigma : {res_fit.lognormal.mu}, {res_fit.lognormal.sigma}")

            raw_ymin = abs(axs[rowi, 0].get_ylim()[0])
            # raw_fit.pdf() -> (x, y)
            raw_ymax = max(np.amax(raw_fit.lognormal.pdf()), np.amax(raw_fit.pdf()[1]))

            res_ymin = abs(axs[rowi, 1].get_ylim()[0])
            # res_fit.pdf() -> (x, y)
            res_ymax = max(np.amax(res_fit.lognormal.pdf()), np.amax(res_fit.pdf()[1]))

            # set yaxis max value adjsut to lognormal and empirical values
            axs[rowi, 0].set_ylim(None, raw_ymax + raw_ymin)
            axs[rowi, 1].set_ylim(None, res_ymax + res_ymin)

            axs[0, 0].set_title('Raw')
            axs[0, 1].set_title('Deseasonlized')

            # disable y label on right side plot
            axs[rowi, 0].set_ylabel(f"PDF - " + rf"${TARGET_MAP[target]}$")
            axs[rowi, 1].yaxis.label.set_visible(False)

            # remove legend title
            for coli in range(2):
                # axs[rowi, coli].legend(loc='best')
                axs[rowi, coli].legend()
                axs[rowi, coli].grid(True, zorder=-5)
                axs[rowi, coli].set_xlabel('')
                axs[rowi, coli].set_xscale('linear')
                axs[rowi, coli].set_yscale('linear')
                axs[rowi, coli].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1E'))
                axs[rowi, coli].annotate(multipanel_labels[rowi, coli], (-0.08, 1.05), xycoords='axes fraction',
                                fontsize='medium', fontweight='bold')

                # small yticks
                for tick in axs[rowi, coli].yaxis.get_major_ticks():
                    tick.label.set_fontsize('small')
                for tick in axs[rowi, coli].xaxis.get_major_ticks():
                    tick.label.set_fontsize('small')

        fig.tight_layout()
        output_prefix = f'{station_name}_fitpdf'
        png_path = output_dir / (output_prefix + '.png')
        svg_path = output_dir / (output_prefix + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close(fig)


def plot_pdf(targets=['PM10', 'PM25'], sample_size=48, output_size=24):
    assert len(targets) > 1
    jongno_fname = 'input_jongno_imputed_hourly_pandas.csv'
    seoul_fname = 'input_seoul_imputed_hourly_pandas.csv'

    sns.set_context('paper')
    sns.set_palette('tab10')

    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    stations = ['종로구']

    # plot distribution of target (raw data)
    nrows = 2
    ncols = 2
    if nrows == 1 or ncols == 1:
        # 1D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:ncols])
    else:
        # 2D
        multipanel_labels = np.array(list(string.ascii_uppercase)[:(nrows*ncols)]).reshape(nrows, ncols)
    multipanellabel_position = (-0.08, 1.02)

    # rough figure size
    w_pad, h_pad = 0.1, 0.15
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    # ax_size = min(7.22 / ncols, 9.45 / nrows)
    ax_size = min(7.22 / ncols, 9.45 / nrows)
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
        fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

        for rowi, target in enumerate(targets):
            dataset = data.UnivariateRNNMeanSeasonalityDataset(
                            station_name=station_name,
                            target=target,
                            filepath=INPUTDATA_path / seoul_fname,
                            features=[target],
                            fdate=train_valid_fdate,
                            tdate=train_valid_tdate,
                            sample_size=sample_size,
                            output_size=output_size)
            dataset.preprocess()

            df_raw = dataset.ys_raw
            df_res = dataset.ys

            arr_raw = df_raw[target].to_numpy()
            arr_res = df_res[target].to_numpy()
            dist_raw = np.random.normal(np.mean(arr_raw), np.std(arr_raw), len(arr_raw))
            dist_res = np.random.normal(np.mean(arr_res), np.std(arr_res), len(arr_res))
            df_plot_raw = pd.DataFrame.from_dict({target: arr_raw, 'gaussian': dist_raw})
            df_plot_res = pd.DataFrame.from_dict({target: arr_res, 'gaussian': dist_res})
            skews = np.zeros(2)
            skews[0] = sp.stats.skew(df_plot_raw.loc[:, target].to_numpy())
            skews[1] = sp.stats.skew(df_plot_res.loc[:, target].to_numpy())

            # bin size determined by Freedman–Diaconis rule, which is default
            # bw = lambda x: 2 * sp.stats.iqr(x) / np.power(len(x), 1.0/3.0)
            if target == 'PM10':
                binwidth = 100
            else:
                binwidth = 50
            # different color for
            sns.histplot(data=df_plot_raw.loc[:, target], ax=axs[rowi, 0],
                         bins=binwidth, color='#1f77b460', stat='density')
            sns.kdeplot(data=df_plot_raw.loc[:, target], ax=axs[rowi, 0],
                        color='#ff7f0e', # tab10:orange
                        linewidth=2.5)

            sns.histplot(data=df_plot_res.loc[:, target], ax=axs[rowi, 1],
                         bins=binwidth, color='#1f77b460', stat='density')
            sns.kdeplot(data=df_plot_res.loc[:, target], ax=axs[rowi, 1],
                        color='#ff7f0e', # tab10:orange
                        linewidth=2.5)

            axs[0, 0].set_title('Raw')
            axs[0, 1].set_title('Deseasonlized')

            # disable y label on right side plot
            axs[rowi, 0].set_ylabel(f"PDF - " + rf"${TARGET_MAP[target]}$")
            axs[rowi, 1].yaxis.label.set_visible(False)

            # remove legend title
            for coli in range(2):
                # axs[rowi, coli].legend().remove()
                # axs[rowi, coli].grid(True, zorder=-5)
                axs[rowi, coli].set_xlabel('')
                axs[rowi, coli].set_xscale('linear')
                axs[rowi, coli].set_yscale('linear')
                axs[rowi, coli].yaxis.set_ticks([])
                axs[rowi, coli].yaxis.set_ticklabels([])
                # axs[rowi, coli].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1E'))
                axs[rowi, coli].annotate(multipanel_labels[rowi, coli], (-0.04, 1.05), xycoords='axes fraction',
                                fontsize='medium', fontweight='bold')

                axs[rowi, coli].annotate('Skewness: {0:.2f}'.format(skews[coli]),
                                        xy=(0.6, 0.9), xycoords='axes fraction',
                                        bbox=dict(boxstyle="square", fc='white', fill=True, linewidth=0.5),
                                        fontsize='medium')

                # small yticks
                for tick in axs[rowi, coli].yaxis.get_major_ticks():
                    tick.label.set_fontsize('x-small')
                for tick in axs[rowi, coli].xaxis.get_major_ticks():
                    tick.label.set_fontsize('small')

        fig.tight_layout()
        output_prefix = f'{station_name}_pdf'
        png_path = output_dir / (output_prefix + '.png')
        svg_path = output_dir / (output_prefix + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cdf", nargs='*',
        help="plot CDF")
    parser.add_argument("-p", "--pdf", nargs='*',
        help="plot PDF")
    parser.add_argument("-f", "--fitpdf", nargs='*',
        help="plot PDF")
    parser.add_argument("-d", "--ccdf", nargs='*',
        help="plot CCDF")

    args = vars(parser.parse_args())

    targets = ['PM10', 'PM25']
    if args["cdf"] != None:
        plot_cdf(targets=targets, sample_size=48, output_size=24)

    if args["fitpdf"] != None:
        plot_fit_pdf(targets=targets, sample_size=48, output_size=24)

    if args["pdf"] != None:
        plot_pdf(targets=targets, sample_size=48, output_size=24)

    if args["ccdf"] != None:
        plot_ccdf(targets=targets, sample_size=48, output_size=24)

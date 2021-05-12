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

MCCR_RESDIR = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MCCR'
MSE_RESDIR = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MSE'

MCCR_RESDIR_72 = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MCCR_72'
MSE_RESDIR_72 = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MSE_72'

SEOULTZ = pytz.timezone('Asia/Seoul')

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

CASE_DICT = {
    'OU' : 'OU',
    'ARIMA_(2, 0, 0)' : 'AR(2)',
    'ARIMA_(3, 0, 0)' : 'AR(3)',
    'MLPMSUnivariate' : 'MLP (Univariate)',
    'RNNAttentionUnivariate' : 'Attention',
    'XGBoost' : 'XGBoost',
    'MLPMSMultivariate' : 'MLP (Multivariate)',
    'RNNLSTNetSkipMultivariate' : 'LSTNet (Skip)',
    'MLPTransformerMultivariate' : 'TST',
    'MLPMSMCCRUnivariate' : 'MLP (Univariate)',
    'RNNAttentionMCCRUnivariate': 'Attention',
    'MLPMSMCCRMultivariate' : 'MLP (Multivariate)',
    'RNNLSTNetSkipMCCRMultivariate' : 'LSTNet (Skip)',
    'MLPTransformerMCCRMultivariate' : 'TST'
}

def relu(x):
    return np.maximum(x, 0.0)

def get_df(data_path):
    obs_file_name = 'df_test_obs.csv'
    sim_file_name = 'df_test_sim.csv'
    df_obs = pd.read_csv(data_path / obs_file_name, header=0, parse_dates=[0])
    df_sim = pd.read_csv(data_path / sim_file_name, header=0, parse_dates=[0])

    df_obs.set_index('date', inplace=True)
    df_sim.set_index('date', inplace=True)

    # apply relu
    df_obs = df_obs.applymap(relu)
    df_sim = df_sim.applymap(relu)

    return df_obs, df_sim

def find_case(model, target, cases):
    for case in cases[target]:
        if CASE_DICT[case] == model:
            return case

    raise ValueError(f"The model, {model}, doesn't exists")

def plot_horizons(input_dir, output_dir, horizons, model, cases,
        station_name="종로구", target="PM10", loss='MSE',
        output_size=24):
    sns.set_context('paper')
    sns.set_palette('tab10')
    nrows = 2
    ncols = 4
    multipanel_labels = np.array(list(string.ascii_uppercase)[:(nrows*ncols)]).reshape(nrows, ncols)
    multipanellabel_position = (-0.08, 1.02)

    # rough figure size
    w_pad, h_pad = 0.1, 0.30
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    # ax_size = min(7.22 / ncols, 9.45 / nrows)
    ax_size = min(14.44 / ncols, 9.45 / nrows)
    # legend_size = 0.6 * fig_size
    fig_size_w = ax_size*ncols
    fig_size_h = ax_size*nrows

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

    case_name = find_case(model, target, cases)

    fdate = dt.datetime(2019, 2, 16, 0).astimezone(SEOULTZ)
    tdate = dt.datetime(2019, 3, 31, 23).astimezone(SEOULTZ)

    data_path = input_dir / case_name / station_name / target / 'csv'
    df_obs, df_sim = get_df(data_path)

    df_obs_dates = df_obs[fdate:tdate]
    df_sim_dates = df_sim[fdate:tdate]

    for coli, h in enumerate(horizons):

        obs = df_obs[str(h - 1)].to_numpy()
        sim = df_sim[str(h - 1)].to_numpy()

        # rowi == 0 : scatter
        maxval = np.nanmax([np.nanmax(obs), np.nanmax(sim)])
        xs = np.array(list(range(round(maxval))))
        best_fit = np.polyfit(obs, sim, 1)

        axs[0, coli].scatter(obs, sim, color="tab:blue", alpha=0.8, s=(4.0,))
        # base line
        axs[0, coli].plot(xs, xs, color="black", alpha=0.7, linewidth=1)
        # best fit line
        axs[0, coli].plot(xs, xs * best_fit[0] + best_fit[1], color="tab:orange", alpha=0.7, linewidth=1.5)
        axs[0, coli].annotate(r'$y = {{{0:.2f}}} x + {{{1:.2f}}}$'.format(
                    best_fit[0], best_fit[1]),
                    xy=(0.35, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="square", fc=None, fill=False, linewidth=0.3),
                    fontsize='medium')
        axs[0, coli].set_aspect(1.0)

        axs[0, coli].annotate(multipanel_labels[0, coli], (-0.08, 1.05), xycoords='axes fraction',
                            fontsize='medium', fontweight='bold')
        axs[0, coli].set_title(f"{h}h", {
            'fontsize': 'large'
        })
        axs[0, coli].set_xlim([0.0, maxval*1.1])
        axs[0, coli].set_ylim([0.0, maxval*1.1])

        axs[0, coli].set_xlabel('actual')

        if coli == 0:
            axs[0, coli].set_ylabel('predicted')
        else:
            axs[0, coli].set_ylabel('')

        for tick in axs[0, coli].xaxis.get_major_ticks():
            tick.label.set_fontsize('x-small')
        for tick in axs[0, coli].yaxis.get_major_ticks():
            tick.label.set_fontsize('x-small')

        # rowi == 1 : line

        # filter by date
        _dates = df_obs.index + dt.timedelta(hours=h)
        dates = pd.date_range(fdate, tdate, freq='1H', tz=SEOULTZ)

        obs_dates = df_obs_dates.loc[:, str(h - 1)].to_numpy()
        sim_dates = df_sim_dates.loc[:, str(h - 1)].to_numpy()
        maxval = np.nanmax([np.nanmax(obs_dates), np.nanmax(sim_dates)])
        axs[1, coli].plot(dates, obs_dates, color="tab:blue", linewidth=1.0, alpha=0.8, label="target")
        axs[1, coli].plot(dates, sim_dates, color="tab:orange", linewidth=1.5, alpha=0.8, label="predicted")

        axs[1, coli].xaxis.set_major_locator(
            mdates.MonthLocator(interval=1, tz=SEOULTZ))
        axs[1, coli].xaxis.set_major_formatter(
            mdates.DateFormatter('%Y/%m', tz=SEOULTZ))

        axs[1, coli].annotate(multipanel_labels[1, coli], (-0.08, 1.05), xycoords='axes fraction',
                            fontsize='medium', fontweight='bold')
        axs[1, coli].set_ylim([0.0, maxval*1.1])
        axs[1, coli].set_xlabel('date')
        axs[1, coli].legend()

        if coli == 0:
            axs[1, coli].set_ylabel(r'$' + TARGET_MAP[target] + r'$')
        else:
            axs[1, coli].set_ylabel('')

        for tick in axs[1, coli].xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')
        for tick in axs[1, coli].yaxis.get_major_ticks():
            tick.label.set_fontsize('x-small')

    fig.tight_layout()
    output_prefix = f'{station_name}_{target}_{loss}_{model}_horizons'
    png_path = output_dir / (output_prefix + '.png')
    svg_path = output_dir / (output_prefix + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close(fig)

def plot_mse(station_name="종로구", targets=["PM10", "PM25"], sample_size=48, output_size=24):
    cases_PM10 = ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate',
        'XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']
    cases_PM25 = ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate',
        'XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']

    output_dir = SCRIPT_DIR / ('out' + str(sample_size)) / 'horizon_mse'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    if sample_size == 72:
        input_dir = MSE_RESDIR_72
    else:
        input_dir = MSE_RESDIR

    cases = {
        'PM10': cases_PM10,
        'PM25': cases_PM25
    }
    # horizons = [1, 4, 8, 24]
    horizons = [3, 6, 12, 24]
    model = 'TST'
    models = ['LSTNet (Skip)', 'TST']

    for target in targets:
        for model in models:
            plot_horizons(input_dir, output_dir, horizons, model, cases,
                station_name=station_name, target=target, loss='MSE',
                output_size=output_size)

def plot_mccr(station_name="종로구", targets=["PM10", "PM25"], sample_size=48, output_size=24):
    cases_PM10 = ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate',
        'XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']
    cases_PM25 = ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate',
        'XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']

    output_dir = SCRIPT_DIR / ('out' + str(sample_size)) / 'horizon_mccr'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    if sample_size == 72:
        input_dir = MCCR_RESDIR_72
    else:
        input_dir = MCCR_RESDIR

    cases = {
        'PM10': cases_PM10,
        'PM25': cases_PM25
    }
    # horizons = [1, 4, 8, 24]
    horizons = [3, 6, 12, 24]
    # horizons = [1, 4, 8, 24]
    horizons = [3, 6, 12, 24]
    models = ['LSTNet (Skip)', 'TST']

    for target in targets:
        for model in models:
            plot_horizons(input_dir, output_dir, horizons, model, cases,
                station_name=station_name, target=target, loss='MCCR',
                output_size=output_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", required=True, nargs='?',
        default='mse', help="set method")
    parser.add_argument("-n", "--name", nargs='?',
        type=str, default='종로구', help="station_name")
    parser.add_argument("-s", "--samples", nargs='?',
        type=int, default=48, help="sample size")

    targets = ["PM10", "PM25"]

    args = vars(parser.parse_args())
    if args['name']:
        station_name = str(args['name'])
    else:
        station_name = '종로구'
    sample_size = int(args["samples"])

    if args["method"] == 'mse':
        plot_mse(station_name=station_name, targets=targets, sample_size=sample_size, output_size=24)
    else:
        plot_mccr(station_name=station_name, targets=targets, sample_size=sample_size, output_size=24)




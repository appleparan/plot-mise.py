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

def plot(is_precompute=False):
    sns.set_context('paper')
    sns.set_palette('tab10')

    jongno_fname = 'input_jongno_imputed_hourly_pandas.csv'
    seoul_fname = 'input_seoul_imputed_hourly_pandas.csv'

    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    stations = ['종로구']
    targets = ['PM10', 'PM25']
    nlags = 7

    # Seasonality
    nrows = 2
    ncols = 5
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
            plot_data_dir = SCRIPT_DIR / 'out' / 'sea' / station_name / target / 'csv'
            Path.mkdir(plot_data_dir, parents=True, exist_ok=True)
            plot_png_dir = SCRIPT_DIR / 'out' / 'sea' / station_name / target / 'png'
            Path.mkdir(plot_png_dir, parents=True, exist_ok=True)
            plot_svg_dir = SCRIPT_DIR / 'out' / 'sea' / station_name / target / 'svg'
            Path.mkdir(plot_svg_dir, parents=True, exist_ok=True)
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
            if is_precompute == False:
                dataset.plot_seasonality(plot_data_dir, plot_png_dir, plot_svg_dir)
                continue

            df_raw = dataset.ys_raw
            df_res = dataset.ys

            data_path = plot_data_dir
            fdate = dt.datetime(2016, 1, 1, 0).astimezone(SEOULTZ)
            tdate = dt.datetime(2016, 12, 31, 23).astimezone(SEOULTZ)
            df_raw_2016 = df_raw[(df_raw.index >= fdate) & (df_raw.index <= tdate)]
            df_res_2016 = df_res[(df_res.index >= fdate) & (df_res.index <= tdate)]
            df_sea_year = pd.read_csv(data_path / 'annual_seasonality_20160101_20161231.csv',
                                        index_col=[0],
                                        parse_dates=['date'])
            df_sea_week = pd.read_csv(data_path / 'weekly_seasonality_20160307_20160313.csv',
                                        index_col=[0],
                                        parse_dates=['day'])
            df_sea_hour = pd.read_csv(data_path / 'hourly_seasonality_2016050100_2016050123.csv',
                                        index_col=[0],
                                        parse_dates=['hour'])

            # axs[rowi, 0] == raw values
            sns.lineplot(data=df_raw_2016, ax=axs[rowi, 0], legend=False, linewidth=0.5)

            # smoothing is true
            # axs[rowi, 1] == yearly seasonality
            if 'ys_smooth' in df_sea_year.columns:
                # convert wide-form dataframe to long-form
                _df_sea_year = df_sea_year.reset_index()
                _df_sea_year = _df_sea_year.rename(columns={'index': 'date'})
                sns.lineplot(x='date', y='value', hue='variable',
                    data=pd.melt(_df_sea_year, ['date']), ax=axs[rowi, 1],
                    palette=['#1f76b4', '#d62728'], linewidth=1.0)

                # thicken yearly seasonality with smoothing
                axs[rowi, 1].lines[1].set_linewidth(2.0)
                # # set yearly seasonality with smoothing 'tab10:red
                # axs[rowi, 1].lines[1].set_color('#d62728')
                # lighten yearly seasonality without smoothing
                axs[rowi, 1].lines[0].set_alpha(0.6)

                # add legend
                leg_handles, leg_labels = axs[rowi, 1].get_legend_handles_labels()
                dic = {'ys_vanilla': 'raw', 'ys_smooth': 'smoothed'}
                leg_labels = [dic.get(l, l) for l in leg_labels]
                # remove legend title
                axs[rowi,1].legend(leg_handles, leg_labels,
                                fancybox=True,
                                fontsize='small')
            else:
                sns.lineplot(x='date', y='ys',
                                data=df_sea_year, ax=axs[rowi, 1], legend=False)

            # axs[rowi, 2] == weekly seasonality
            sns.lineplot(data=df_sea_week, ax=axs[rowi, 2], legend=False)
            # axs[rowi, 3] == hourly seasonality
            sns.lineplot(data=df_sea_hour, ax=axs[rowi, 3], legend=False)
            # axs[rowi, 4] == residuals
            sns.lineplot(data=df_res_2016, ax=axs[rowi, 4], legend=False, linewidth=0.5)

            axs[rowi, 0].xaxis.set_major_locator(
                mdates.MonthLocator(bymonth=[1, 4, 7, 10], tz=SEOULTZ))
            axs[rowi, 0].xaxis.set_minor_locator(
                mdates.MonthLocator(bymonth=range(1, 13), tz=SEOULTZ))
            axs[rowi, 0].xaxis.set_major_formatter(
                mdates.DateFormatter('%m', tz=SEOULTZ))

            axs[rowi, 1].xaxis.set_major_locator(
                mdates.MonthLocator(bymonth=[1, 4, 7, 10], tz=SEOULTZ))
            axs[rowi, 1].xaxis.set_minor_locator(
                mdates.MonthLocator(bymonth=range(1, 13), tz=SEOULTZ))
            axs[rowi, 1].xaxis.set_major_formatter(
                mdates.DateFormatter('%m', tz=SEOULTZ))

            axs[rowi, 2].xaxis.set_major_locator(
                mdates.HourLocator(byhour=0, tz=SEOULTZ))
            axs[rowi, 2].xaxis.set_major_formatter(
                mdates.DateFormatter('%a', tz=SEOULTZ))

            axs[rowi, 3].xaxis.set_major_locator(
                mdates.HourLocator(byhour=[0, 4, 8, 12, 16, 20], tz=SEOULTZ))
            axs[rowi, 3].xaxis.set_minor_locator(
                mdates.HourLocator(tz=SEOULTZ))
            axs[rowi, 3].xaxis.set_major_formatter(
                mdates.DateFormatter('%H', tz=SEOULTZ))

            axs[rowi, 4].xaxis.set_major_locator(
                mdates.MonthLocator(bymonth=[1, 4, 7, 10], tz=SEOULTZ))
            axs[rowi, 4].xaxis.set_minor_locator(
                mdates.MonthLocator(bymonth=range(1, 13), tz=SEOULTZ))
            axs[rowi, 4].xaxis.set_major_formatter(
                mdates.DateFormatter('%m', tz=SEOULTZ))

            if rowi != 0:
                axs[rowi, 0].set_xlabel('month', fontsize='small')
                axs[rowi, 1].set_xlabel('month', fontsize='small')
                axs[rowi, 2].set_xlabel('weekday', fontsize='small')
                axs[rowi, 3].set_xlabel('hour', fontsize='small')
                axs[rowi, 4].set_xlabel('month', fontsize='small')
            else:
                for coli in range(ncols):
                    axs[rowi, coli].set_xlabel('')

            for coli in range(ncols):
                if coli == 0:
                    axs[rowi, coli].set_ylabel(r'$\mathrm{{{0:s}}}$'.format(
                        TARGET_MAP[target]), fontsize='medium')
                else:
                    axs[rowi, coli].set_ylabel('')

            for coli in range(ncols):
                axs[rowi, coli].set_title("")
                #axs[i].set_ylabel(target, fontsize='small')
                axs[rowi, coli].xaxis.grid(True, visible=True, which='major')
                for tick in axs[rowi, coli].xaxis.get_major_ticks():
                    tick.label.set_fontsize('small')
                for tick in axs[rowi, coli].yaxis.get_major_ticks():
                    tick.label.set_fontsize('small')

                axs[rowi, coli].annotate(multipanel_labels[rowi, coli], (-0.08, 1.02), xycoords='axes fraction',
                                fontsize='large', fontweight='bold')

            # dataset.plot_seasonality(plot_data_dir / target, plot_png_dir / target, plot_svg_dir / target)
        fig.tight_layout()
        output_fname = f"{station_name}_sea"
        png_path = output_dir / (output_fname + '.png')
        svg_path = output_dir / (output_fname + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--precompute", action='store_true',
        default='False', help="using precomputed seasonality, if not, just compute seasonality")

    args = vars(parser.parse_args())

    is_precompute = args['precompute']
    plot(is_precompute=is_precompute)

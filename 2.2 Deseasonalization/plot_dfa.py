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
    'u': r'\mathrm{\mathsf{Wind Speed (Zonal)}}',
    'v': r'\mathrm{\mathsf{Wind Speed (Meridional)}}',
    'pres': r'\mathrm{\mathsf{Pressure}}',
    'humid': r'\mathrm{\mathsf{Relative Humidity}}',
    'prep': r'\mathrm{\mathsf{Rainfall}}',
    'snow': r'\mathrm{\mathsf{Snow}}'
}

def plot():
    jongno_fname = 'input_jongno_imputed_hourly_pandas.csv'
    seoul_fname = 'input_seoul_imputed_hourly_pandas.csv'

    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    stations = ['종로구']
    targets = ['PM10', 'PM25']

    # 1 2
    # 3 4
    # PM10: 1 2
    # PM25: 3 4
    # 1 3: DFA2 (raw)
    # 2 4: DFA2 (normed)
    nrows = len(targets)
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

    order = 2

    def model_func(x, A, B):
        return A * np.power(x, B)

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
        coef_annot = np.zeros((nrows, ncols))
        gamma_annot = np.zeros((nrows, ncols))

        for rowi, target in enumerate(targets):
            dfa_dir = SCRIPT_DIR / 'DFA' / station_name / target

            df_res = pd.read_csv(dfa_dir / (f'DFA_res_o{order}.csv'))
            df_res_norm = pd.read_csv(dfa_dir / (f'DFA_norm_res_o{order}.csv'))

            df = pd.read_csv(dfa_dir / (f'DFA_o{order}.csv'))
            df_norm = pd.read_csv(dfa_dir / (f'DFA_norm_o{order}.csv'))

            if rowi == nrows-1:
                for coli in range(ncols):
                    axs[rowi, coli].set_xlabel(r'lag $s$ (hour)', fontsize='small')

            q_list = df.columns[1:]
            lag = df['s'].to_numpy()
            large_s = int(len(lag) * 0.3)

            ## fitted line for large s
            sns.lineplot(x='s', y='value', hue='q',
                            data=pd.melt(df, id_vars=['s'], var_name='q'),
                            ax = axs[rowi, 0])
            ## plot h(2) = 1/2
            base_lines = 10.0**(-2) * np.power(lag, 0.5)
            axs[rowi, 0].plot(lag, base_lines,
                        label=r'$h(2) = 0.5$',
                        alpha=0.7, color='tab:gray', linestyle='dashed')

            ## plot fitted line
            p0 = (1., 1.e-5)
            popt, pcov = sp.optimize.curve_fit(model_func, lag[large_s:], df.to_numpy()[:, -1][large_s:], p0)
            coef_annot[rowi, 0] = popt[1]
            gamma_annot[rowi, 0] = 2.0 * (1.0 - popt[1])
            estimated = model_func(lag, popt[0], popt[1])
            axs[rowi, 0].plot(lag, estimated,
                        label=r'$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$'.format(
                            popt[1], gamma_annot[rowi, 0]),
                        alpha=0.7, color='tab:cyan', linestyle='dashed')
            # annotate
            # axs[rowi, 0].annotate(r'$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$'.format(
            #                 coef_annot[rowi, 0], gamma_annot[rowi, 0]),
            #                 xy=(lag[1], estimated[1]),
            #                 xycoords='data',
            #                 xytext=(-20, 60), textcoords='offset points',
            #                 bbox=dict(boxstyle="round", fc='white', linewidth=0.3),
            #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", linewidth=0.3),
            #                 fontsize='small')
            print(f"{target} (raw) ({multipanel_labels[rowi, 0]}) - h(2): {coef_annot[rowi, 0]}, gamma: {gamma_annot[rowi, 0]}")
            leg_handles, leg_labels = axs[rowi, 0].get_legend_handles_labels()
            leg_labels[0] = r'$h(2)$'
            axs[rowi, 0].get_legend().remove()
            #axs[rowi, 0].legend(leg_handles, leg_labels, fontsize='x-small')
            axs[rowi, 0].set_xscale('log')
            axs[rowi, 0].set_yscale('log')

            # res
            q_list = df_res.columns[1:]
            lag = df_res['s'].to_numpy()
            large_s = int(len(lag) * 0.3)
            ## fitted line
            sns.lineplot(x='s', y='value', hue='q',
                            data=pd.melt(df_res, id_vars=['s'], var_name='q'),
                            ax = axs[rowi, 1])
            ## plot h(2) = 1/2
            base_lines = 10.0**(-2) * np.power(lag, 0.5)
            axs[rowi, 1].plot(lag, base_lines,
                        label=r'$h(2) = 0.5$',
                        alpha=0.7, color='tab:gray', linestyle='dashed')
            ## plot fitted lin
            p0 = (1., 1.e-5)
            popt, pcov = sp.optimize.curve_fit(model_func, lag[large_s:], df_res.to_numpy()[:, -1][large_s:], p0)
            coef_annot[rowi, 1] = popt[1]
            gamma_annot[rowi, 1] = 2.0 * (1.0 - popt[1])
            estimated = model_func(lag, popt[0], popt[1])
            axs[rowi, 1].plot(lag, estimated,
                        label=r'$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$'.format(
                            popt[1], gamma_annot[rowi, 1]),
                        alpha=0.7, color='tab:cyan', linestyle='dashed')
            # axs[rowi, 1].annotate(r'$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$'.format(
            #                 coef_annot[rowi, 1], gamma_annot[rowi, 1]),
            #                 xy=(lag[1], estimated[1]),
            #                 xycoords='data',
            #                 xytext=(-20, 30), textcoords='offset points',
            #                 bbox=dict(boxstyle="round", fc='white', linewidth=0.3),
            #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", linewidth=0.3),
            #                 fontsize='small')
            print(f"{target} (norm) ({multipanel_labels[rowi, 1]}) - h(2): {coef_annot[rowi, 1]}, gamma: {gamma_annot[rowi, 1]}")
            leg_handles, leg_labels = axs[rowi, 1].get_legend_handles_labels()
            leg_labels[0] = r'$h(2)$'
            axs[rowi, 1].get_legend().remove()

            axs[rowi, 1].set_xscale('log')
            axs[rowi, 1].set_yscale('log')

            # set ylabel only for left plot
            axs[rowi, 0].set_ylabel(r'$F(s)$' + f" (${TARGET_MAP[target]}$)")
            axs[rowi, 1].set_ylabel('')
            # set xlabel only for bottom plot
            if rowi == len(targets)-1:
                axs[rowi, 0].set_xlabel(r'$s$')
                axs[rowi, 1].set_xlabel(r'$s$')
            else:
                axs[rowi, 0].set_xlabel('', fontsize='medium')
                axs[rowi, 1].set_xlabel('', fontsize='medium')
        
        x_mins = np.zeros((len(targets), 2))
        x_maxs = np.zeros((len(targets), 2))
        y_mins = np.zeros((len(targets), 2))
        y_maxs = np.zeros((len(targets), 2))

        for (rowi, coli) in itertools.product(range(len(targets)), range(2)):
            x_mins[rowi, coli], x_maxs[rowi, coli] = axs[rowi, coli].get_xlim()
            y_mins[rowi, coli], y_maxs[rowi, coli] = axs[rowi, coli].get_ylim()

        for (rowi, coli) in itertools.product(range(len(targets)), range(2)):
            axs[rowi, coli].set_title("")

            axs[rowi, coli].xaxis.grid(True, visible=True, which='major')
            for tick in axs[rowi, coli].xaxis.get_major_ticks():
                tick.label.set_fontsize('small')
            for tick in axs[rowi, coli].yaxis.get_major_ticks():
                tick.label.set_fontsize('small')

            tot_max = np.amax([np.amax(x_maxs[rowi, :]), np.amax(y_maxs[rowi, :])])
            tot_xmin = np.power(10, np.floor(np.log10(np.amin(x_mins[rowi, :]))))
            tot_ymin = np.power(10, np.floor(np.log10(np.amin(y_mins[rowi, :]))))
            axs[rowi, coli].set_xlim(tot_xmin, axs[rowi, coli].get_xlim()[1])
            axs[rowi, coli].set_ylim(tot_ymin, axs[rowi, coli].get_ylim()[1])
            axs[rowi, coli].annotate(multipanel_labels[rowi, coli], multipanellabel_position, xycoords='axes fraction',
                                fontsize='large', fontweight='bold')

        output_fname = f"{station_name}_DFA2"
        png_path = output_dir / (output_fname + '.png')
        svg_path = output_dir / (output_fname + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

if __name__ == '__main__':
    plot()

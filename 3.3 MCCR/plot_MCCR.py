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

SCRIPT_DIR = Path(__file__).parent.absolute()

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"

INPUTDATA_path = SCRIPT_DIR / '..' / 'data'
SEOULTZ = timezone('Asia/Seoul')

def plot():
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    nx = 121
    x = np.linspace(-6, 6, num=nx)
    sigmas = [0.6, 0.8, 1.1]
    styles = [
        {
            'color': 'tab:blue',
            'linestyle': 'solid',
            'annote_x': 4.5,
            'ytext': 4,
        },
        {
            'color': 'tab:orange',
            'linestyle': 'dashed',
            'annote_x': -4.0,
            'ytext': 4,
        },
        {
            'color': 'tab:green',
            'linestyle': 'dashdot',
            'annote_x': 4.5,
            'ytext': -20,
        }
    ]
    ax_size = min(7.22, 5.415)

    fig, axs = plt.subplots(1, 1,
        figsize=(7.22, 5.415),
        dpi=600,
        frameon=False,
        subplot_kw={
            'clip_on': False,
    })

    for sigma, style in zip(sigmas, styles):
        def mccr(x):
            return sigma**2 * (1.0 - np.exp(-x**2 / sigma**2))

        y = np.array(list(map(mccr, x)))
        ycoord = mccr(style['annote_x'])

        axs.plot(x, y, color=style['color'], linestyle=style['linestyle'], linewidth=3)
        axs.annotate(r'$\alpha\ =\ $' + f'{sigma:.1f}',
                    xy=(style['annote_x'], ycoord),
                    xytext=(0, style['ytext']),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize='x-large')
        axs.set_ylabel(r'$l_\alpha(Y_t - \hat{Y}_t)$', {'fontsize': 'x-large'})
        axs.set_xlabel(r'$Y_t - \hat{Y}_t$', {'fontsize': 'x-large'})

        for tick in axs.xaxis.get_major_ticks():
                tick.label.set_fontsize('large')
        for tick in axs.yaxis.get_major_ticks():
                tick.label.set_fontsize('large')
    fig.tight_layout()
    output_fname = f"loss_MCCR"
    png_path = output_dir / (output_fname + '.png')
    svg_path = output_dir / (output_fname + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()

if __name__ == '__main__':
    plot()
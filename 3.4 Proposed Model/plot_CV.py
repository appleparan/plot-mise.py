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

    dates = [
        {
            'train_fdate': '2008/01/01',
            'train_tdate': '2009/12/31',
            'valid_fdate': '2010/01/01',
            'valid_tdate': '2010/06/30'
        },
        {
            'train_fdate': '2010/07/01',
            'train_tdate': '2012/06/30',
            'valid_fdate': '2012/07/01',
            'valid_tdate': '2012/12/31'
        },
        {
            'train_fdate': '2013/01/01',
            'train_tdate': '2014/12/31',
            'valid_fdate': '2015/01/01',
            'valid_tdate': '2015/06/30'
        },
        {
            'train_fdate': '2015/07/01',
            'train_tdate': '2017/12/31',
            'valid_fdate': '2018/01/01',
            'valid_tdate': '2018/12/31'
        },
    ]

    test_fdate = mdates.date2num(dt.datetime.strptime('2019/01/01', "%Y/%m/%d"))
    test_tdate = mdates.date2num(dt.datetime.strptime('2020/10/31', "%Y/%m/%d"))

    xlims = (mdates.date2num(dt.datetime.strptime('2007/10/01', "%Y/%m/%d")),
        mdates.date2num(dt.datetime.strptime('2021/01/31', "%Y/%m/%d")))

    ax_size = min(7.22, 5.415)

    fig, axs = plt.subplots(1, 1,
        figsize=(7.22, 5.415),
        dpi=600,
        frameon=False,
        subplot_kw={
            'clip_on': False,
    })

    locators = []
    for i_d, dict_d in enumerate(dates):
        train_fdate = mdates.date2num(dt.datetime.strptime(dict_d['train_fdate'], "%Y/%m/%d"))
        train_tdate = mdates.date2num(dt.datetime.strptime(dict_d['train_tdate'], "%Y/%m/%d"))
        valid_fdate = mdates.date2num(dt.datetime.strptime(dict_d['valid_fdate'], "%Y/%m/%d"))
        valid_tdate = mdates.date2num(dt.datetime.strptime(dict_d['valid_tdate'], "%Y/%m/%d"))
        duration_train = train_tdate - train_fdate
        duration_valid = valid_tdate - valid_fdate
        if i_d == 0:
            train_label = 'train'
            valid_label = 'valid'
        else:
            train_label = None
            valid_label = None
        axs.barh(i_d + 1, duration_train, left=train_fdate,
                align='center', height=.25, color='tab:blue', label=train_label)
        axs.barh(i_d + 1, duration_valid, left=valid_fdate,
                align='center', height=.25, color='tab:orange', label=valid_label)
        # skip valid_tdate because train_fdate will be almost same as last valid_tdate
        locators = locators + [train_fdate, valid_fdate]

    axs.barh(len(dates) + 1, test_tdate - test_fdate,  left=test_fdate,
                align='center', height=.25, color='tab:green', label='test')
    locators = locators + [test_fdate, test_tdate]

    axs.legend(fontsize='x-large')
    # axs.legend().remove()
    axs.xaxis.set_major_locator(mticker.FixedLocator(locators))
    axs.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y/%m', tz=SEOULTZ))
    axs.invert_yaxis()
    axs.grid(b=True, axis='x')

    axs.set_xlim(xlims)
    axs.set_xlabel('dates', {'fontsize': 'x-large'})
    axs.set_ylabel('block', {'fontsize': 'x-large'})

    labels = [mdates.num2date(tick).strftime("%Y/%m") for tick in axs.get_xticks()]
    axs.set_xticklabels(labels, rotation=70)

    for tick in axs.xaxis.get_major_ticks():
        tick.label.set_fontsize('medium')
    for tick in axs.yaxis.get_major_ticks():
        tick.label.set_fontsize('large')
    
    plt.subplots_adjust(bottom=0.15)
    fig.tight_layout()

    # fig.tight_layout()
    output_fname = f"CV"
    png_path = output_dir / (output_fname + '.png')
    svg_path = output_dir / (output_fname + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()

if __name__ == '__main__':
    plot()
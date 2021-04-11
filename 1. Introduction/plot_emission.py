import argparse
import glob
import itertools
from pathlib import Path
import re
import string

import pandas as pd
import numpy as np
import scipy as sp

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.metrics
import sklearn

import statsmodels.api as sm
import statsmodels.tsa.stattools as tsast
import statsmodels.graphics.tsaplots as tpl

import bokeh
from bokeh.models import Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png, export_svgs
from bokeh.palettes import Category10, Category20

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import mpl_toolkits.axes_grid1.parasite_axes as paxes

SCRIPT_DIR = Path(__file__).parent.absolute()
EPSILON = 1e-10

# plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"

# Korean
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = "Noto Sans CJK KR"

def plot_(data_dir, output_dir, target='PM10'):
    filename = target + '_emission_sources.csv'
    df = pd.read_csv(data_dir / filename)
    
    fig = plt.figure(figsize=(7.2, 7.2), dpi=600)
    ax = fig.gca()

    years = [str(i) for i in (range(2011, 2018))]
    emissions = df.iloc[:, 0]

    cmap = plt.get_cmap('tab20')
    colors = [mcolors.rgb2hex(c) for c in cmap.colors]
    # swap 
    colors[11], colors[1] = colors[1], colors[11]
    colors[2], colors[3] = colors[3], colors[2]

    ax.stackplot(years, df[years].to_numpy().astype(np.int64),
                 labels=emissions, colors=colors)
    ax.legend(loc='upper left')

    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize('small')
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize('medium')

    fig.tight_layout()
    sns.despine()
    output_to_plot = "emissions_" + target
    png_path = output_dir / (output_to_plot + '.png')
    svg_path = output_dir / (output_to_plot + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()

def plot():
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR / 'data'

    plot_(data_dir, output_dir, target='PM10')
    plot_(data_dir, output_dir, target='PM25')

if __name__ == '__main__':
    plot()

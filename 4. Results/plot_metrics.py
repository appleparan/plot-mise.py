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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.metrics
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

# suppress SettingsWithCopy Warning
pd.set_option('mode.chained_assignment', None)

# matplotlib params
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"

MCCR_RESDIR = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MCCR'
MSE_RESDIR = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MSE'

MCCR_RESDIR_72 = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MCCR_72'
MSE_RESDIR_72 = SCRIPT_DIR / '..' / '..' / 'Figures_DATA_MSE_72'

INPUTDATA_path = SCRIPT_DIR / '..' / 'data'
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

def load_df(input_dir: Path, case_name: str,
    station_name='종로구', target='PM10'):

    data_path = input_dir / case_name / station_name / target / 'csv'

    obs_file_name = 'df_test_obs.csv'
    sim_file_name = 'df_test_sim.csv'
    df_obs = pd.read_csv(data_path / obs_file_name, header=0)
    df_sim = pd.read_csv(data_path / sim_file_name, header=0)

    df_obs.set_index('date', inplace=True)
    df_sim.set_index('date', inplace=True)

    return df_obs, df_sim

def compute_quantiles(_df):
    # q1,q2,q3 have time (0~23) as row and q value as column (single column)
    q1 = _df.quantile(q=0.25)
    q2 = _df.quantile(q=0.5)
    q3 = _df.quantile(q=0.75)

    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    iqr.columns = ['iqr']
    upper.columns = ['upper']
    lower.columns = ['lower']

    q1.rename('q1', inplace=True)
    q2.rename('q2', inplace=True)
    q3.rename('q3', inplace=True)
    iqr.rename('iqr', inplace=True)
    upper.rename('upper', inplace=True)
    lower.rename('lower', inplace=True)

    return {
        'q1': q1,
        'q2': q2,
        'q3': q3,
        'iqr': iqr,
        'upper': upper,
        'lower': lower
    }

def outliers(df, q):
    def _outliers(sr):
        """
        Each row applied
        """
        return sr[(sr > q['upper'][sr.name]) | (sr < q['lower'][sr.name])]

    return df.apply(_outliers, axis='index').dropna()

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(np.mean(np.square(_error(actual, predicted))))

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))

def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))

def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))

def r2(actual: np.ndarray, predicted: np.ndarray):
    return sklearn.metrics.r2_score(actual, predicted)

def pcorr(actual: np.ndarray, predicted: np.ndarray):
    """ Pearson Correlation Coefficient """
    pcorr, p_val = sp.stats.pearsonr(actual, predicted)
    return pcorr, p_val

def scorr(actual: np.ndarray, predicted: np.ndarray):
    """ Spearman Rank Correlation Coefficient """
    scorr, p_val = sp.stats.spearmanr(actual, predicted)
    return scorr, p_val

def corr(actual: np.ndarray, predicted: np.ndarray):
    """ Correlation Coefficient from LSTNet paper  """
    avg_m = np.mean(predicted)
    avg_o = np.mean(actual)

    diff_a = actual - avg_o
    diff_p = predicted - avg_m

    numerator = np.dot(diff_a, diff_p)
    denominator = np.sqrt(np.sum(np.square(diff_a)) * np.sum(np.square(diff_p)))

    return np.mean(np.divide(numerator, denominator))

def fb(actual: np.ndarray, predicted: np.ndarray):
    """ Fractional Bias """
    avg_a = np.mean(actual)
    avg_p = np.mean(predicted)
    return 2.0 * ((avg_a - avg_p) / (avg_a + avg_p + np.finfo(float).eps))

def fae(actual: np.ndarray, predicted: np.ndarray):
    """ Fractional Absolute Error """
    avg_a = np.mean(actual)
    avg_p = np.mean(predicted)
    return 2.0 * (np.mean(np.fabs(predicted - actual)) / (avg_a + avg_p + np.finfo(float).eps))

def nmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Mean Square Error """
    return np.mean(np.square(actual - predicted)) / (np.mean(actual) * np.mean(predicted) + np.finfo(float).eps)

def mg(actual: np.ndarray, predicted: np.ndarray):
    """ Geometric Mean Bias """
    return np.exp(np.mean(np.log(actual + 1.0)) - np.mean(np.log(predicted + 1.0)))

def vg(actual: np.ndarray, predicted: np.ndarray):
    """ Geometric Variance """
    return np.exp(np.mean(np.square(np.log(actual + 1.0) - np.log(predicted + 1.0))))

def fac2(actual: np.ndarray, predicted: np.ndarray):
    """ The fraction of predictions within a factor of two of observations """
    frac = predicted / actual
    return ((0.5 <= frac) & (frac <= 2.0)).sum() / len(predicted)

def smape(actual: np.ndarray, predicted: np.ndarray):
    """ Symmetric Mean Absolute Percentage Error
    Reference:
        * Wambura, Stephen, Jianbin Huang, and He Li. "Long-range forecasting in feature-evolving data streams." Knowledge-Based Systems 206 (2020): 106405.
        * Tofallis, Chris. "A better measure of relative prediction accuracy for model selection and model estimation." Journal of the Operational Research Society 66.8 (2015): 1352-1362.
    """
    return np.mean(np.divide(np.abs(actual - predicted), (np.abs(actual) + np.abs(predicted) + np.finfo(float).eps) * 0.5))

def mnfb(actual: np.ndarray, predicted: np.ndarray):
    """Mean Normalized Factor Bias, B_MNFB

    MNFB = mean(F)
    F = M/O - 1 where M >= O
        1 - O/M where M < O

    Reference:
    * Yu, Shaocai, et al. "New unbiased symmetric metrics for evaluation of air quality models." Atmospheric Science Letters 7.1 (2006): 26-34.
    """
    df = pd.DataFrame.from_dict({
        'actual': actual,
        'predicted': predicted,
    })
    df['err'] = df['predicted'] - df['actual']
    df['mo'] = df['predicted'] / df['actual']
    df['om'] = df['actual'] / df['predicted']

    df_pos = df.loc[df['err'] >= 0, :]
    df_neg = df.loc[df['err'] < 0, :]

    # drop inf
    df_pos.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_pos.dropna(inplace=True)
    df_neg.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_neg.dropna(inplace=True)

    # df_pos == M >= O -> sum(M/O - 1)
    # df_neg == M < O  -> sum(1 - O/M)
    return sum(df_pos['mo'] - 1.0) / len(df_pos['mo']) + \
           sum(1.0 - df_neg['om']) / len(df_neg['om'])

def mnafe(actual: np.ndarray, predicted: np.ndarray):
    """Mean Normalized Absolute Factor Error

    MNGFE = mean(|F|)
    F = M/O - 1 where M >= O
        1 - O/M where M < O

    Reference:
    * Yu, Shaocai, et al. "New unbiased symmetric metrics for evaluation of air quality models." Atmospheric Science Letters 7.1 (2006): 26-34.
    """
    df = pd.DataFrame.from_dict({
        'actual': actual,
        'predicted': predicted,
    })
    df['err'] = df['predicted'] - df['actual']
    df['mo'] = df['predicted'] / df['actual']
    df['om'] = df['actual'] / df['predicted']

    df_pos = df.loc[df['err'] >= 0, :]
    df_neg = df.loc[df['err'] < 0, :]

    # drop inf
    df_pos.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_pos.dropna(inplace=True)
    df_neg.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_neg.dropna(inplace=True)

    # df_pos == M >= O -> sum(M/O - 1)
    # df_neg == M < O  -> sum(1 - O/M)
    return sum(np.fabs(df_pos['mo'] - 1.0)) / len(df_pos['mo']) + \
           sum(np.fabs(1.0 - df_neg['om'])) / len(df_neg['om'])

def nmbf(actual: np.ndarray, predicted: np.ndarray):
    """Normalized Mean Bias Factor

    MNFB = mean(|F|)
    F = M/O - 1 where M >= O
        1 - O/M where M < O

    Reference:
    * Yu, Shaocai, et al. "New unbiased symmetric metrics for evaluation of air quality models." Atmospheric Science Letters 7.1 (2006): 26-34.
    """
    avg_m = np.mean(predicted)
    avg_o = np.mean(actual)

    # df_pos == M >= O -> avg(M)/avg(O) - 1
    # df_neg == M < O  -> 1 - avg(O)/avg(M)
    if avg_m >= avg_o:
        return avg_m / avg_o - 1
    else:
        return 1 - avg_o / avg_m

def nmaef(actual: np.ndarray, predicted: np.ndarray):
    """Normalized Mean Absolute Error Factor

    MNFB = mean(|F|)
    F = M/O - 1 where M >= O
        1 - O/M where M < O

    Reference:
    * Yu, Shaocai, et al. "New unbiased symmetric metrics for evaluation of air quality models." Atmospheric Science Letters 7.1 (2006): 26-34.
    """
    mage = np.mean(np.fabs(predicted - actual))
    avg_m = np.mean(predicted)
    avg_o = np.mean(actual)

    # df_pos == M >= O -> avg(M)/avg(O) - 1
    # df_neg == M < O  -> 1 - avg(O)/avg(M)
    if avg_m >= avg_o:
        return mage / avg_o
    else:
        return mage / avg_m

def ioa(actual: np.ndarray, predicted: np.ndarray):
    """Index of Agreement

    MNFB = mean(|F|)
    F = M/O - 1 where M >= O
        1 - O/M where M < O

    Reference:
    * Yu, Shaocai, et al. "New unbiased symmetric metrics for evaluation of air quality models." Atmospheric Science Letters 7.1 (2006): 26-34.
    """
    mage = np.mean(np.square(predicted - actual))
    avg_o = np.mean(actual)
    mao = np.fabs(predicted - avg_o)
    oao = np.fabs(actual - avg_o)

    # df_pos == M >= O -> avg(M)/avg(O) - 1
    # df_neg == M < O  -> 1 - avg(O)/avg(M)
    return 1.0 - mage / np.mean(np.square(mao + oao))

def compute_metric(df_obs, df_sim, metric, output_size=24):
    """
    metric should be one of 'MSE', 'MAE', and 'MAPE'
    available options
        PCORR : Pearson Correlation Coef.
        SCORR : Spearman's Rank Correlation Coef.

    """

    if metric == 'PCORR' or metric == 'SCORR' or metric == 'R2' or metric == 'CORR':
        res = np.zeros(output_size + 1)
        p_val = np.zeros(output_size + 1)
        res[0] = 1.0
        lags = list(range(output_size + 1))
    else:
        res = np.zeros(output_size)
        p_val = np.zeros(output_size)
        lags = list(range(1, output_size + 1))

    for i in range(output_size):
        if metric == 'PCORR':
            res[i + 1], p_val[i + 1] = pcorr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'SCORR':
            res[i + 1], p_val[i + 1] = scorr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'CORR':
            res[i + 1] = corr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'R2':
            res[i + 1] = r2(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MSE':
            res[i] = mse(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'RMSE':
            res[i] = rmse(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MAE':
            res[i] = mae(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MAPE':
            res[i] = mape(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MAAPE':
            res[i] = maape(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'NMSE':
            res[i] = nmse(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MG':
            res[i] = mg(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'VG':
            res[i] = vg(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'FB':
            res[i] = fb(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'FAE':
            res[i] = fae(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'FAC2':
            res[i] = fac2(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'SMAPE':
            res[i] = smape(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MNFB':
            res[i] = mnfb(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MNAFE':
            res[i] = mnafe(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'NMBF':
            res[i] = nmbf(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'NMAEF':
            res[i] = nmaef(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'IOA':
            res[i] = ioa(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        else:
            raise NameError("this metric ( ",metric, ") doesn't supported : ")

    return lags, res, p_val


def plot_metric(metric, cases, input_dir = Path('.'), output_dir = Path('.'),
                station_name='종로구', targets=['PM10', 'PM25'], loss='MSE', sample_size=48, output_size=24):
    """Plot metric to 2x2 subplots
        |   Univariate  | Multivariate
    ----------------------------------
        |               |
    PM10|               |
        |               |
    ----------------------------------
        |               |
    PM25|               |
        |               |
    """
    sns.set_context("paper")
    sns.color_palette("tab10")
    nrows = 2
    ncols = 2
    multipanel_labels = np.array(list(string.ascii_uppercase)[:nrows * ncols]).reshape(nrows, ncols)

    # rough figure size
    w_pad, h_pad = 1.08, 1.08
    # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
    legend_size = 1.0 / 72 * 10.0 * 10
    ax_size = min(7.22 / ncols, (9.45 - legend_size) / nrows)
    # legend_size = 0.6 * fig_size
    fig_size_w = ax_size*ncols
    fig_size_h = ax_size*nrows + legend_size
    legend_frac = legend_size / (fig_size_h + legend_size)

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ax_size*ncols, ax_size*nrows + legend_size),
                            dpi=600,
                            frameon=False,
                            subplot_kw={
                                'clip_on': False,
                                'box_aspect': 1
                            })
    fig.tight_layout(w_pad=w_pad, h_pad=h_pad)
    fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

    # targets is actually useless

    df_10 = pd.DataFrame()
    df_25 = pd.DataFrame()
    dfs = {
        'PM10': df_10,
        'PM25': df_25,
    }

    for target in targets:
        for case in cases[target]['Univariate'] + cases[target]['Multivariate']:
            df_obs, df_sim = load_df(input_dir, case,
                                     station_name=station_name, target=target)

            lags, res, p_val = compute_metric(df_obs, df_sim, metric)

            if dfs[target].empty:
                dfs[target].loc[:, 'time'] = lags

            dfs[target].loc[:, case] = res


    for rowi, target in enumerate(targets):
        # Univariate
        cases_uni = cases[target]['Univariate']
        df_uni = dfs[target].loc[:, cases_uni]
        df_uni.loc[:, 'time'] = dfs[target].loc[:, 'time'].to_numpy()
        sns.lineplot(x='time', y='value', hue='variable',
                     data=pd.melt(df_uni, ['time']),
                     ax = axs[rowi, 0])

        # Multivariate
        cases_mul = cases[target]['Multivariate']
        df_mul = dfs[target].loc[:, cases_mul]
        df_mul.loc[:, 'time'] = dfs[target].loc[:, 'time'].to_numpy()
        sns.lineplot(x='time', y='value', hue='variable',
                     data=pd.melt(df_mul, ['time']),
                     ax = axs[rowi, 1])

        ymins = np.zeros((2, 2))
        ymaxs = np.zeros((2, 2))
        for coli in range(2):
            ymins[rowi, coli], ymaxs[rowi, coli] = axs[rowi, coli].get_ylim()

        for coli in range(2):
            # legend
            if rowi == 0:
                leg_handles, _leg_labels = axs[rowi, coli].get_legend_handles_labels()
                # new legend label map
                leg_labels = [CASE_DICT[l] for l in _leg_labels]
                axs[rowi, coli].legend(leg_handles, leg_labels,
                                    bbox_to_anchor=(0.0, 1.02, 1, legend_frac),
                                    ncol=1,
                                    mode='expand',
                                    loc='lower left',
                                    borderaxespad=0.5,
                                    fancybox=True,
                                    fontsize='medium')
            else:
                axs[rowi, coli].get_legend().remove()

            # xaxis
            axs[rowi, coli].set_xlabel('time', fontsize='medium')
            axs[rowi, coli].set_xlim(-0.5, 24.5)

            # multipanel lagbel
            axs[rowi, coli].annotate(multipanel_labels[rowi, coli],
                                    (-0.08, 1.05), xycoords='axes fraction',
                                    fontsize='large', fontweight='bold')

            # xticks
            axs[rowi, coli].xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
            axs[rowi, coli].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

            # xgrid
            axs[rowi, coli].xaxis.grid(True, visible=True, which='major')

            for tick in axs[rowi, coli].xaxis.get_major_ticks():
                tick.label.set_fontsize('medium')
            for tick in axs[rowi, coli].yaxis.get_major_ticks():
                tick.label.set_fontsize('medium')

            # hide x label except last row
            if rowi == 0:
                axs[rowi, coli].xaxis.label.set_visible(False)

            # hide y label except first col
            if coli == 1:
                axs[rowi, coli].yaxis.label.set_visible(False)

            # yaxis
            # set limit of y axis
            axs[rowi, coli].set_ylabel(metric, fontsize='medium')

            # ylims
            axs[rowi, coli].set_ylim(np.amin(ymins[rowi, :]), np.amax(ymaxs[rowi, :]))

            if metric == 'MAPE':
                # Best MAPE => 1.0
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'MAAPE':
                # Best MAAPE => 0.0, Worst => pi/2
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'MAE':
                # Best MAE => Lower
                pass
            elif metric == 'MSE':
                # Best MSE => Lower
                pass
            elif metric == 'R2':
                # Best R2 => 1.0, Worst - 1.0
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'PCORR':
                # Best PCORR => 1.0
                axs[rowi, coli].set_ylabel(r"Pearson's $r$", fontsize='medium')
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'SCORR':
                # Best SCORR => 1.0
                axs[rowi, coli].set_ylabel(r"Spearman's $\rho$", fontsize='x-small')
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'CORR':
                # Best SCORR => 1.0
                axs[rowi, coli].set_ylabel("Corr. Coef.", fontsize='x-small')
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'FB':
                # Best FB => 0.0
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'NMSE':
                # Best NMSE => 0.0
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'MG':
                # Best MG => 1.0, Worst => far from 1.0
                pass
            elif metric == 'VG':
                # Best VG => 1.0, Worst => far from 1.0
                pass
            elif metric == 'FAC2':
                # Best FAC2 => 1.0
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'SMAPE':
                # Best SMAPE => 1.0
                axs[rowi, coli].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
                axs[rowi, coli].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif metric == 'MNFB':
                axs[rowi, coli].set_ylim(-3, 3)
            elif metric == 'MNAFE':
                axs[rowi, coli].set_ylim(0, 3)

            # ygrid
            axs[rowi, coli].yaxis.grid(True, visible=True, which='major')

        axs[rowi, 0].set_ylabel(f'{metric} - ${TARGET_MAP[target]}$', fontsize='medium')
    fig.subplots_adjust(left=0.13, top=1.02-legend_frac)

    output_prefix = f'{station_name}_{metric}'
    png_path = output_dir / (output_prefix + '.png')
    svg_path = output_dir / (output_prefix + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close(fig)


def plot_metrics_mse(station_name='종로구', targets=['PM10', 'PM25'], sample_size=48, output_size=24):
    cases = {
        'PM10': {
            'Univariate': ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']
        },
        'PM25': {
            'Univariate': ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']
        }
    }

    output_dir = SCRIPT_DIR / ('out' + str(sample_size)) / 'metric_mse'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    if sample_size == 72:
        input_dir = MSE_RESDIR_72
    else:
        input_dir = MSE_RESDIR

    metrics = [ 'MSE', 'MAE', 'MAPE', 'NMSE',
                'PCORR', 'SCORR', 'CORR', 'R2',
                'FB', 'FAE', 'MG', 'VG',
                'FAC2', 'MAAPE', 'SMAPE', 'IOA',
                'MNFB', 'MNAFE', 'NMBF', 'NMAEF']

    for metric in metrics:
        print(f"MSE Loss - {metric} - {station_name} - {sample_size}")
        plot_metric(metric, cases,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    station_name=station_name,
                    targets=targets,
                    sample_size=sample_size,
                    output_size=output_size,
                    loss='MSE')

def plot_metrics_mccr(station_name='종로구', targets=['PM10', 'PM25'], sample_size=48, output_size=24):
    cases = {
        'PM10': {
            'Univariate': ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']
        },
        'PM25': {
            'Univariate': ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']
        }
    }

    output_dir = SCRIPT_DIR / ('out' + str(sample_size)) / 'metric_mccr'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    if sample_size == 72:
        input_dir = MCCR_RESDIR_72
    else:
        input_dir = MCCR_RESDIR

    metrics = [ 'MSE', 'MAE', 'MAPE', 'NMSE',
                'PCORR', 'SCORR', 'CORR', 'R2',
                'FB', 'FAE', 'MG', 'VG',
                'FAC2', 'MAAPE', 'SMAPE', 'IOA',
                'MNFB', 'MNAFE', 'NMBF', 'NMAEF']

    for metric in metrics:
        print(f"MCCR Loss - {metric} - {station_name} - {sample_size}")
        plot_metric(metric, cases,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    station_name=station_name,
                    targets=targets,
                    sample_size=sample_size,
                    output_size=output_size,
                    loss='MCCR')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", required=True, nargs='?',
        default='mse', help="set method")
    parser.add_argument("-n", "--name", nargs='?',
        type=str, default='종로구', help="station_name")
    parser.add_argument("-s", "--samples", nargs='?',
        type=int, default=48, help="sample size")

    args = vars(parser.parse_args())

    if args['name']:
        station_name = str(args['name'])
    else:
        station_name = '종로구'

    sample_size = int(args["samples"])

    targets = ['PM10', 'PM25']

    if args["method"] == 'mse':
        plot_metrics_mse(station_name=station_name, targets=targets, sample_size=sample_size, output_size=24)
    else:
        plot_metrics_mccr(station_name=station_name, targets=targets, sample_size=sample_size, output_size=24)

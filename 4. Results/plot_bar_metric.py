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
mpl.rcParams['hatch.linewidth'] = 0.3

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
    """ Mean Squared Error """
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
    """Mean Normalized Factor Bias

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
    res = np.zeros(output_size)
    p_val = np.zeros(output_size)
    lags = list(range(1, output_size + 1))

    for i in range(output_size):
        if metric == 'PCORR':
            res[i], p_val[i] = pcorr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'SCORR':
            res[i], p_val[i] = scorr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'CORR':
            res[i] = corr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'R2':
            res[i] = r2(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
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

def plot_table_loss(cases_mse, cases_mccr, metrics, losses, horizons, input_dirs: dict = {48: Path('.')}, output_dir = Path('.'),
               station_name='종로구', target='PM10', sample_size=72, output_size=24):
    """Create result table like LSTNet gathered by loss function

    1. Table Index Compare loss function)
        * Row : Methods -> Metric
            * Row 1 : methods (cases)
            * Row 2 : metrics
        * Column : loss function -> horizon size
            * Col 1 : loss function
            * Col 2 : horizon
    """
    # targets is actually useless

    # zero index horizon
    horizons0 = np.array(horizons) - 1

    cases_tot = cases_mccr[target]['Univariate'] + cases_mccr[target]['Multivariate']
    cases_tot_names = [CASE_DICT[c] for c in cases_tot]
    rows = list(itertools.product(cases_tot_names, metrics))
    cols = list(itertools.product(losses, horizons))

    mindex = pd.MultiIndex.from_tuples(rows, names=["model", "metric"])
    cindex = pd.MultiIndex.from_tuples(cols, names=["loss", "horizon"])
    df = pd.DataFrame(index=mindex, columns=cindex)
    idx = pd.IndexSlice

    for loss in losses:
        if loss == 'MCCR':
            cases = cases_mccr[target]['Univariate'] + cases_mccr[target]['Multivariate']
        else:
            cases = cases_mse[target]['Univariate'] + cases_mse[target]['Multivariate']

        for case in cases:
            df_obs, df_sim = load_df(input_dirs[loss], case,
                                    station_name=station_name, target=target)

            for metric in metrics:
                lags, res, p_val = compute_metric(df_obs, df_sim, metric)

                # if metric == 'NMAEF':
                #     print(loss, case, metric, res[horizons0])

                # weird bug, can't assign lterables
                df.loc[idx[CASE_DICT[case], metric], idx[loss, :]] = res[horizons0]

    return df

def plot_plot_tables_error(plot_cases, station_name='종로구', targets=['PM10', 'PM25'], sample_size=72, output_size=24):
    """Plot single error metric comparisioon per horizons

    each subplot compare cases of plot_metric
    each subplot indicate single horizon

    x: horizons
        each x: LSTNet- MSE, TST - MSE, LSTNet - MCCR, TST - MCCR
    y: metric value
    --------------
    | metrics[0] |
    --------------
    | metrics[1] |
    --------------
    | metrics[2] |
    --------------
    | metrics[3] |
    --------------
    """
    sns.set_context('paper')
    sns.set_palette('tab10')
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    # metrics = ['RMSE', 'CORR', 'NMBF', 'NMAEF']
    metrics = ['NMAEF', 'RMSE', 'CORR']
    cases_mccr = {
        'PM10': {
            'Univariate': ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']
        },
        'PM25': {
            'Univariate': ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']
        }
    }

    cases_mse = {
        'PM10': {
            'Univariate': ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']
        },
        'PM25': {
            'Univariate': ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']
        }
    }

    losses = ['MSE', 'MCCR']
    horizons = [1, 4, 8, 24]
    cases = ['LSTNet (Skip)', 'TST']

    if sample_size == 48:
        input_dirs = {
            'MSE': MSE_RESDIR,
            'MCCR': MCCR_RESDIR
        }
    else:
        input_dirs = {
            'MSE': MSE_RESDIR_72,
            'MCCR': MCCR_RESDIR_72
        }

    for target in targets:
        print(f"{station_name} - {target} - {sample_size}")
        df = plot_table_loss(cases_mse, cases_mccr, metrics, losses, horizons,
                             input_dirs=input_dirs,
                             output_dir=output_dir,
                             station_name=station_name,
                             target=target,
                             sample_size=sample_size,
                             output_size=output_size)

        nrows = len(metrics)
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

        fig, axs = plt.subplots(nrows, ncols,
            figsize=(7.22*ncols, ax_size*nrows),
            dpi=600,
            frameon=False,
            subplot_kw={
                'clip_on': False,
            })
        # keep right distance between subplots
        # fig.tight_layout(h_pad=h_pad)
        # fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

        idx = pd.IndexSlice
        len_groups = len(losses) * len(cases)

        for rowi, metric in enumerate(metrics):
            # 1, 2, 3, 4 -> 4, 8, 12, 16 -> 2, 6, 10, 14
            width = 0.5
            xs = -2 * width + np.arange(start=2*width+1, stop=(len_groups+2)*len_groups*width, step=(len_groups+2)*width)

            rects, labels = [], []
            # tab10 & tab20
            # https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
            colors = ['tab:blue', 'tab:orange', '#A0CBE8', '#FFBE7D']
            alphas = [1.0, 1.0, 1.0, 1.0]
            hatches = [None, None, '//', '//']
            ymin = np.finfo('float64').max
            ymax = np.finfo('float64').min
            for reci, (loss, case) in enumerate(itertools.product(losses, cases)):
                # 1. MSE - LSTNet - tab:blue
                # 2. MSE - TST - tab:purple
                # 3. MCCR - LSTNet - tab:orange
                # 4. MCCR - TST - tab:red
                ys = df.loc[idx[case, metric], idx[loss, :]]
                ymin = min(ymin, np.amin(ys))
                ymax = max(ymax, np.amax(ys))
                print(f"{loss} - {case}", ymax, np.amax(ys))
                label = f"{loss} - {case}"
                base = -1.5 * width
                offset = reci * width
                rects.append(axs[rowi].bar(xs + base + offset, ys, label=label, \
                             width=width, color=colors[reci],
                             zorder=4,
                             alpha=alphas[reci], hatch=hatches[reci]))

            # scale ylimit
            mag = abs(ymax - ymin)
            nymin = max(0.0, ymin - mag * 0.15)
            nymax = ymax + mag * 0.1
            axs[rowi].set_ylim(nymin, nymax)

            if rowi == len(metrics) - 1:
                axs[rowi].set_xlabel('horizon')
            axs[rowi].set_ylabel(metric)
            axs[rowi].yaxis.grid(True)
            axs[rowi].set_xticks(xs)
            axs[rowi].set_xticklabels(horizons)
            axs[rowi].legend(fontsize='small')

        fig.tight_layout(h_pad=h_pad)
        # fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)
        output_prefix = f'{station_name}_{target}_metrics_bar'
        png_path = output_dir / (output_prefix + '.png')
        svg_path = output_dir / (output_prefix + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close(fig)


def plot_plot_tables_bias(plot_cases, station_name='종로구', targets=['PM10', 'PM25'], sample_size=72, output_size=24):
    """Plot bias for violin plot

    * Each axes is target (PM10, PM2.5)
    * Draw split violin plot, split is done by loss
    * Each split violin is single model
    --------------
    | PM10  |
    --------------
    | PM2.5 |
    --------------
    """
    sns.set_context('paper')
    sns.set_palette('tab10')
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    # metrics = ['RMSE', 'CORR', 'NMBF', 'NMAEF']
    metrics = ['NMAEF', 'RMSE', 'CORR']
    cases_mccr = {
        'PM10': {
            'Univariate': ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']
        },
        'PM25': {
            'Univariate': ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSMCCRUnivariate', 'RNNAttentionMCCRUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMCCRMultivariate', 'RNNLSTNetSkipMCCRMultivariate', 'MLPTransformerMCCRMultivariate']
        }
    }

    cases_mse = {
        'PM10': {
            'Univariate': ['OU', 'ARIMA_(2, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']
        },
        'PM25': {
            'Univariate': ['OU', 'ARIMA_(3, 0, 0)', 'MLPMSUnivariate', 'RNNAttentionUnivariate'],
            'Multivariate': ['XGBoost', 'MLPMSMultivariate', 'RNNLSTNetSkipMultivariate', 'MLPTransformerMultivariate']
        }
    }

    losses = ['MSE', 'MCCR']
    horizons = [1, 4, 8, 24]

    if sample_size == 48:
        input_dirs = {
            'MSE': MSE_RESDIR,
            'MCCR': MCCR_RESDIR
        }
    else:
        input_dirs = {
            'MSE': MSE_RESDIR_72,
            'MCCR': MCCR_RESDIR_72
        }

    nrows = len(targets)
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

    for hi, horizon in enumerate(horizons):
        cols = list(itertools.product(losses, plot_cases))
        cindex = pd.MultiIndex.from_tuples(cols, names=["loss", "case"])
        idx = pd.IndexSlice

        fig, axs = plt.subplots(nrows, ncols,
            figsize=(7.22*ncols, ax_size*nrows),
            dpi=600,
            frameon=False,
            subplot_kw={
                'clip_on': False,
        })
        # keep right distance between subplots
        # fig.tight_layout(h_pad=h_pad)
        # fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)
        for rowi, target in enumerate(targets):
            print(f"{station_name} - {target} - {str(horizon).zfill(2)}")
            df_fb = pd.DataFrame(columns=cindex)
            for loss in losses:
                if loss == 'MCCR':
                    cases = cases_mccr[target]['Univariate'] + cases_mccr[target]['Multivariate']
                else:
                    cases = cases_mse[target]['Univariate'] + cases_mse[target]['Multivariate']

                for case in cases:
                    case_legend = CASE_DICT[case]
                    if case_legend not in plot_cases:
                        continue

                    df_obs, df_sim = load_df(input_dirs[loss], case,
                                            station_name=station_name, target=target)
                    df_diff = df_sim - df_obs

                    """
                    columns : case, loss, horizon
                    rows : total_length
                    """
                    df_fb.loc[idx[:], idx[loss, case_legend]] = df_diff.loc[:, str(horizon-1)]

            # for annotation
            df_metric = plot_table_loss(cases_mse, cases_mccr, ['NMBF', 'MNFB', 'FB'], losses, horizons,
                                input_dirs=input_dirs,
                                output_dir=output_dir,
                                station_name=station_name,
                                target=target,
                                sample_size=sample_size,
                                output_size=output_size)
            len_groups = len(losses) * len(cases)

            # tab10 & tab20
            # https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
            ymin = np.finfo('float64').max
            ymax = np.finfo('float64').min

            # scale ylimit
            _ymin, _ymax = axs[rowi].get_ylim()
            df_plot = pd.melt(df_fb)

            plot_cases_numeric = {c: i for i, c in enumerate(plot_cases)}
            df_plot['case_num'] = df_plot['case'].map(lambda c: plot_cases_numeric[c])

            sns.violinplot(data=df_plot,
                x="case_num", y='value', hue="loss", split=True,
                linewidth=1.7, saturation=1.0, inner='quartile',
                ax=axs[rowi], zorder=2)

            xtickslocs = axs[rowi].get_xticks()
            ymin, ymax = axs[rowi].get_ylim()
            yoff = ymin + abs(ymax - ymin) * 0.9
            for i, xtickloc in enumerate(xtickslocs):
                case_name = plot_cases[i]
                nmbf_mse = df_metric.loc[idx[case_name, 'NMBF'], idx['MSE', horizon]]
                nmbf_mccr = df_metric.loc[idx[case_name, 'NMBF'], idx['MCCR', horizon]]

                # annotate MSE Loss
                axs[rowi].annotate(
                    '{0: .5f}'.format(nmbf_mse),
                    xy=(xtickloc - 0.08, yoff),
                    xycoords='data',
                    xytext=(0, 0),
                    textcoords='offset points',
                    color='black',
                    bbox=dict(boxstyle="square", fc='white', fill=True, linewidth=0.1),
                    ha='right')

                # annotate MCCR Loss
                axs[rowi].annotate(
                    '{0: .5f}'.format(nmbf_mccr),
                    xy=(xtickloc + 0.08, yoff),
                    xycoords='data',
                    xytext=(0, 0),
                    textcoords='offset points',
                    color='black',
                    bbox=dict(boxstyle="square", fc='white', fill=True, linewidth=0.1),
                    ha='left')


            if rowi == 0:
                axs[rowi].set_xlabel('')
            else:
                axs[rowi].set_xlabel('Model')

            axs[rowi].set_xticklabels(plot_cases)
            axs[rowi].set_ylabel(fr"${TARGET_MAP[target]}$")
            axs[rowi].set_ylabel(r'Bias ($M_i - O_i$)')
            axs[rowi].yaxis.grid(True, zorder=-100)
            axs[rowi].set_axisbelow(True)
            axs[rowi].legend(fontsize='small', loc='lower right')

        fig.tight_layout(h_pad=h_pad)
        # fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9)
        output_prefix = f'{station_name}_{str(horizon).zfill(2)}h_bias_violin'
        png_path = output_dir / (output_prefix + '.png')
        svg_path = output_dir / (output_prefix + '.svg')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", nargs='?',
        type=str, default='종로구', help="station_name")

    parser.add_argument("-e", "--error", action='store_true')
    parser.add_argument("-b", "--bias", action='store_true')
    parser.add_argument("-s", "--sample", nargs='?',
        default='48', help="sample_size")

    args = vars(parser.parse_args())

    if args['name']:
        station_name = str(args['name'])
    else:
        station_name = '종로구'

    targets = ['PM10', 'PM25']

    if args["error"]:
        sample_size = int(args["sample"])
        plot_cases = ['LSTNet (Skip)', 'TST']
        plot_plot_tables_error(plot_cases, station_name=station_name, targets=targets, sample_size=sample_size, output_size=24)

    if args["bias"]:
        sample_size = int(args["sample"])
        plot_cases = ['Attention', 'MLP (Multivariate)', 'LSTNet (Skip)', 'TST']
        plot_plot_tables_bias(plot_cases, station_name=station_name, targets=targets, sample_size=sample_size, output_size=24)



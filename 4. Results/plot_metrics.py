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

def relu(x):
    return np.maximum(x, 0.0)

def get_df(base_dir, case_name,
    station_name='종로구', target='PM10'):

    data_path = SCRIPT_DIR.joinpath(case_name, station_name, target, 'csv')

    obs_file_name = 'df_test_obs.csv'
    sim_file_name = 'df_test_sim.csv'
    df_obs = pd.read_csv(data_path / obs_file_name, header=0)
    df_sim = pd.read_csv(data_path / sim_file_name, header=0)

    df_obs.set_index('date', inplace=True)
    df_sim.set_index('date', inplace=True)

    # apply relu
    df_obs = df_obs.applymap(relu)
    df_sim = df_sim.applymap(relu)

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

def fb(actual: np.ndarray, predicted: np.ndarray):
    """ Fractional Bias """
    avg_a = np.mean(actual)
    avg_p = np.mean(predicted)
    return 2.0 * ((avg_a - avg_p) / (avg_a + avg_p + np.finfo(float).eps))

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

def compute_metric(df_obs, df_sim, metric, output_size=24):
    """
    metric should be one of 'MSE', 'MAE', and 'MAPE'
    available options
        PCORR : Pearson Correlation Coef.
        SCORR : Spearman's Rank Correlation Coef.

    """

    if metric == 'PCORR' or metric == 'SCORR' or metric == 'R2':
        res = np.zeros(output_size + 1)
        p_val = np.zeros(output_size + 1)
        res[0] = 1.0
    else:
        res = np.zeros(output_size)
        p_val = np.zeros(output_size)

    for i in range(output_size):
        if metric == 'PCORR':
            res[i + 1], p_val[i + 1] = pcorr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'SCORR':
            res[i + 1], p_val[i + 1] = scorr(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'R2':
            res[i + 1] = r2(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'MSE':
            res[i] = mse(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
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
        elif metric == 'FAC2':
            res[i] = fac2(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        elif metric == 'SMAPE':
            res[i] = smape(df_obs.loc[:, str(i)].to_numpy(), df_sim.loc[:, str(i)].to_numpy())
        else:
            raise NameError("this metric ( ",metric, ") doesn't supported : ")

    return res, p_val

def plot_metrics_mse(station_name='종로구'):
    case_dict = {
        'OU' : 'OU'
        'ARIMA_(2, 0, 0)' : 'AR(2)',
        'MLPMSUnivariate' : 'MLP (Univariate)', 
        'RNNAttentionUnivariate': 'Attention',
        'XGBoost': 'XGBoost',
        'MLPMSMultivariate' : 'MLP (Multivariate)'
        'RNNLSTNetSkipMultivariate': 'LSTNet (Skip)',
        'MLPTransformerSEMultivariate' : 'TST'
    }
    
    # Because univariate models are not ready, skip
    case_dict = {
        'OU' : 'OU'
        'ARIMA_(2, 0, 0)' : 'AR(2)',
        'MLPMSUnivariate' : 'MLP (Univariate)', 
        'RNNAttentionUnivariate': 'Attention',
        'XGBoost': 'XGBoost',
        'MLPMSMCCRMultivariate' : 'MLP (Multivariate)'
        'RNNLSTNetSkipMCCRMultivariate': 'LSTNet (Skip)',
        'MLPTransformerMCCRMultivariate' : 'TST'
    }

    metrics = [ 'MSE', 'MAE', 'MAPE', 'NMSE',
                'PCORR', 'SCORR', 'R2',
                'FB', 'MG', 'VG', 'FAC2',
                'MAAPE', 'SMAPE']

    targets = ['PM10', 'PM25']

def plot_metrics_mccr(station_name='종로구'):
    case_dict = {
        'OU' : 'OU'
        'ARIMA_(2, 0, 0)' : 'AR(2)',
        'MLPMSUnivariate' : 'MLP (Univariate)', 
        'RNNAttentionUnivariate': 'Attention',
        'XGBoost': 'XGBoost',
        'MLPMSMCCRMultivariate' : 'MLP (Multivariate)'
        'RNNLSTNetSkipMCCRMultivariate': 'LSTNet (Skip)',
        'MLPTransformerMCCRMultivariate' : 'TST'
    }
    metrics = [ 'MSE', 'MAE', 'MAPE', 'NMSE',
                'PCORR', 'SCORR', 'R2',
                'FB', 'MG', 'VG', 'FAC2',
                'MAAPE', 'SMAPE']
    targets = ['PM10', 'PM25']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", required=True, nargs=1,
        default='mse', help="set method")
    parser.add_argument("-s", "--mse", nargs='*',
        help="plot MSE")

    args = vars(parser.parse_args())

    # machine learning
    if args["mccr"] != None:
        plot_metrics_mccr(station_name='종로구')

    if args["mse"] != None:
        plot_metrics_mse(station_name='종로구')

    if args["method"] == 'mse':
        for target in targets:
            plot_metrics_mse(station_name='종로구', target=target, sample_size=sample_size, output_size=24)
    else:
        for target in targets:
            plot_metrics_mccr(station_name='종로구', target=target, sample_size=sample_size, output_size=24)






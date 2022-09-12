import sys
sys.path.insert(0, '.')

import argparse

from pathlib import Path

import math
import numpy as np
import pandas as pd

from multiprocessing import Pool
from functools import partial


def taylor_coefs_centered_window(loc_data,
                                 target_var,
                                 taylor_degree=1,
                                 window=21):
    half_window = (window - 1) // 2
    shift_varnames = []
    for l in range(-half_window, half_window + 1):
        if l < 0:
            shift_varname = target_var + '_m' + str(abs(l))
        else:
            shift_varname = target_var + '_p' + str(abs(l))
        
        shift_varnames.append(shift_varname)
        loc_data[shift_varname] = loc_data[[target_var]].shift(-l)
    
    if taylor_degree == 1:
        one_taylor_X = np.concatenate(
            [
                np.ones((window, 1)),
                np.expand_dims(np.arange(window) - half_window, -1)
            ],
            axis = 1
        )
    elif taylor_degree == 2:
        one_taylor_X = np.concatenate(
            [
                np.ones((window, 1)),
                np.expand_dims(np.arange(window) - half_window, -1),
                np.expand_dims(+0.5 * (np.arange(window) - half_window)**2, -1)
            ],
            axis = 1
        )
    
    y = loc_data[shift_varnames].values.astype('float64')
    
    beta_hat = np.linalg.lstsq(one_taylor_X, np.transpose(y), rcond=None)[0]
    
    # clean up beginning and end, where there was not enough data
    # fit to sub-window with fully observed data
    for i in range(half_window):
        beta_hat[:, i] = np.linalg.lstsq(one_taylor_X[(half_window - i):, :],
                                         y[i, (half_window - i):],
                                         rcond=None)[0]
        beta_hat[:, -(i+1)] = np.linalg.lstsq(one_taylor_X[:(half_window + i + 1), :],
                                         y[-(i + 1), :(half_window + i + 1)],
                                         rcond=None)[0]
    
    for d in range(taylor_degree + 1):
        loc_data[target_var + '_taylor_' + str(d)] = beta_hat[d, :]
    
    loc_data = loc_data.drop(shift_varnames, axis=1)
    
    return loc_data


def count_zeros(values):
    return sum(values == 0.)


def prop_zeros(values):
    # print("new call")
    # print(values)
    # print(sum(values == 0.))
    # print(len(values))
    return sum(values == 0.) / len(values)


def clean_outliers(values):
    result = values.copy()
    
    # set all trailing zeros to nan; will be filled with the last nonzero value later
    num_trailing_zeros = len(result.values) - np.max(np.nonzero(values.values)) - 1
    if num_trailing_zeros > 0:
        result[-num_trailing_zeros:] = np.nan
    
    # count number of zero values in a centered rolling 7 day window
    # fill in zero counts at end with last value
    zero_counts = result.rolling(7, center = True).apply(count_zeros)
    zero_counts = zero_counts.fillna(method='ffill')
    
    zero_props = result.rolling(7, center = True).apply(prop_zeros)
    zero_props = zero_props.fillna(method='ffill')
    
    # if more than 2 days of zeros in rolling window, replace with rolling mean
    inds_to_replace = (zero_counts > 2) & (zero_props < 1.0)
    replace_vals = result.rolling(7, center = True).mean()
    replace_vals = replace_vals.fillna(method='ffill')
    result[inds_to_replace] = replace_vals[inds_to_replace]
    
    # if 1 or 2 days of zeros in rolling window, set to nan
    inds_to_replace = (((zero_counts > 0) & (zero_counts <= 2)) | zero_props == 1.0) & (result == 0.)
    result[inds_to_replace] = np.nan
    
    # detect outliers as rolling median +/- 3 IQR, set to nan
    rm = result.rolling(15, center = True, min_periods=1).median()
    resids = result - rm
    riqr = resids.rolling(15, center = True, min_periods=1).quantile(0.75) - \
        resids.rolling(15, center = True, min_periods=1).quantile(0.25)
    lower = (rm - 3 * riqr).fillna(method = 'ffill')
    upper = (rm + 3 * riqr).fillna(method = 'ffill')
    is_outlr = (result < lower) | (result > upper)
    result[is_outlr] = np.nan
    
    # linearly interpolate nan's (corresponding to zeros and detected outliers)
    result.interpolate(inplace=True, limit_direction='both')
    
    return result


def transform_loc_data(loc, data):
    # subset to data for loc
    data = data[data.location == loc].copy()
    
    # data transform and outlier cleaning
    data['corrected_case_rate'] = data['case_rate']
    data.corrected_case_rate[data.corrected_case_rate <= 0] = 0.0
    data.corrected_case_rate = clean_outliers(data.corrected_case_rate)
    
    data['case_rate_sqrt'] = data['case_rate']
    data.case_rate_sqrt[data.case_rate_sqrt <= 0] = 0.0
    data.case_rate_sqrt = np.sqrt(data.case_rate_sqrt)
    data['corrected_case_rate_sqrt'] = data['case_rate_sqrt']
    data.corrected_case_rate_sqrt = clean_outliers(data.corrected_case_rate_sqrt)
    
    data['case_rate_fourthrt'] = data['case_rate']
    data.case_rate_fourthrt[data.case_rate_fourthrt <= 0] = 0.0
    data.case_rate_fourthrt = np.power(data.case_rate_fourthrt, 0.25)
    data['corrected_case_rate_fourthrt'] = data['case_rate_fourthrt']
    data.corrected_case_rate_fourthrt = clean_outliers(data.corrected_case_rate_fourthrt)
    
    # add smooth of cases
    data = data.groupby('location').apply(taylor_coefs_centered_window, target_var = 'corrected_case_rate')
    data = data.groupby('location').apply(taylor_coefs_centered_window, target_var = 'corrected_case_rate_sqrt')
    data = data.groupby('location').apply(taylor_coefs_centered_window, target_var = 'corrected_case_rate_fourthrt')
    
    return data




if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="hierarchicalGP")
    parser.add_argument("--forecast_date", nargs="?", default='2022-04-18', type = str)
    parser.add_argument("--source", nargs="?", default='jhu', type = str)
    args = parser.parse_args()
    forecast_date = args.forecast_date
    source = args.source
    # forecast_date = '2022-04-18'
    # source = 'cdc'
    
    # load data
    data = pd.read_csv(f'data/{source}_data_cached_{forecast_date}.csv')
    data.date = pd.to_datetime(data.date)
    # data = pd.read_csv(f'data/cdc_data_cached_{forecast_date}.csv')
    # data = pd.read_csv(f'data/nyt_data_cached_{forecast_date}.csv')
    
    with Pool(processes=18) as pool:
        results = pool.map(partial(transform_loc_data, data=data), data.location.unique())
    
    results = pd.concat(results, axis = 0)
    results.to_csv(f'data/{source}_data_smoothed_{forecast_date}.csv', index=False)
    # results.to_csv('data/nyt_data_smoothed_2022-04-18.csv', index=False)


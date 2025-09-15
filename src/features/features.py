import pandas as pd
import numpy as np


def add_rolling_and_lags(df, windows=[5, 10], lags=[1, 3], drop_originals=True):
    
    df = df.copy()
    df = df.sort_values(['engine_number', 'cycle']).reset_index(drop=True)
    
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    
    # rolling stats
    for window in windows:
        for sensor in sensor_cols:
            df[f"{sensor}_mean_{window}"] = df.groupby('engine_number')[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f"{sensor}_std_{window}"] = df.groupby('engine_number')[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
    
    # lag features
    for lag in lags:
        for sensor in sensor_cols:
            df[f"{sensor}_lag_{lag}"] = df.groupby('engine_number')[sensor].shift(lag)
    
    # drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)
    
    if drop_originals:
        df = df.drop(columns=sensor_cols)
    
    print(df.shape)
    return df
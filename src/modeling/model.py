import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


def scale_features(df, scaler=None, is_fit=False):
    df = df.copy()
    exclude_cols = ['engine_number', 'cycle', 'RUL_capped']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if is_fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df, scaler
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
        return df, scaler


def split_by_engines(df, train_engines=70, val_engines=15):
    # split by engine not by rows - avoids data leakage
    unique_engines = sorted(df['engine_number'].unique())
    
    train_ids = unique_engines[:train_engines]
    val_ids = unique_engines[train_engines:train_engines + val_engines]
    
    df_train = df[df['engine_number'].isin(train_ids)].copy()
    df_val = df[df['engine_number'].isin(val_ids)].copy()
    
    print(f"train: {len(df_train)} rows, val: {len(df_val)} rows")
    return df_train, df_val


def get_xy(df):
    exclude = ['engine_number', 'cycle', 'RUL_capped']
    feature_cols = [col for col in df.columns if col not in exclude]
    X = df[feature_cols].copy()
    y = df['RUL_capped'].copy() if 'RUL_capped' in df.columns else None
    return X, y


def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred}


def save_model(model, filename, models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print("saved to", filepath)


def load_model(filename, models_dir="models"):
    filepath = os.path.join(models_dir, filename)
    return joblib.load(filepath)
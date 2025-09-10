import pandas as pd
import os

columns = (
    ["engine_number", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_raw(file_name):
    df = pd.read_csv(
        file_name,
        sep=r'\s+',
        header=None,
        names=columns,
        engine='python'
    )
    print(df.shape)
    return df

def drop_low_variance(df, threshold=0.001):
    cols = [col for col in df.columns if col not in ["engine_number", "cycle"]]
    std_vals = df[cols].std()
    low_var = std_vals[std_vals < threshold].index.tolist()
    print("dropping:", low_var)
    return df.drop(columns=low_var)

def add_rul(df, max_rul=130):
    max_cycles = df.groupby("engine_number")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_number", "max_cycle"]
    df = df.merge(max_cycles, on="engine_number")
    df["RUL_capped"] = (df["max_cycle"] - df["cycle"]).clip(upper=max_rul)
    df = df.drop(columns=["max_cycle"])
    return df

def basic_preprocess_train(file_name, variance_threshold=0.001, max_rul=130, save_path=None):
    df = load_raw(file_name)
    df = drop_low_variance(df, threshold=variance_threshold)
    df = add_rul(df, max_rul=max_rul)
    if save_path:
        df.to_csv(save_path, index=False)
        print("saved to", save_path)
    return df

def basic_preprocess_test(file_name, variance_threshold=0.001, save_path=None):
    df = load_raw(file_name)
    df = drop_low_variance(df, threshold=variance_threshold)
    if save_path:
        df.to_csv(save_path, index=False)
    return df
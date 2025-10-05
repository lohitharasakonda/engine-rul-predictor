from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model and scaler
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

LOW_VARIANCE_COLS = [
    "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19",
    "setting_2", "setting_3",
]

FEATURE_COLUMNS = ["setting_1"]
for window in [5, 10]:
    for sensor in SENSORS:
        FEATURE_COLUMNS.append(f"{sensor}_mean_{window}")
        FEATURE_COLUMNS.append(f"{sensor}_std_{window}")
for lag in [1, 3]:
    for sensor in SENSORS:
        FEATURE_COLUMNS.append(f"{sensor}_lag_{lag}")

MAX_RUL = 130


def engineer_features(df):
    df = df.copy()
    df = df.sort_values(["engine_number", "cycle"]).reset_index(drop=True)

    for col in LOW_VARIANCE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    available_sensors = [s for s in SENSORS if s in df.columns]

    for window in [5, 10]:
        for sensor in available_sensors:
            df[f"{sensor}_mean_{window}"] = df.groupby("engine_number")[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f"{sensor}_std_{window}"] = df.groupby("engine_number")[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

    for lag_val in [1, 3]:
        for sensor in available_sensors:
            df[f"{sensor}_lag_{lag_val}"] = df.groupby("engine_number")[sensor].shift(lag_val)

    df = df.drop(columns=available_sensors)
    df = df.dropna().reset_index(drop=True)

    return df


def get_health(rul):
    score = min(100.0, max(0.0, (rul / MAX_RUL) * 100))
    if score >= 60:
        status = "Healthy"
    elif score >= 30:
        status = "Warning"
    else:
        status = "Critical"
    return {"health_score": round(score, 1), "status": status}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode()))

        if "engine_number" not in df.columns:
            df["engine_number"] = 1
        if "cycle" not in df.columns:
            df["cycle"] = range(1, len(df) + 1)

        # run feature engineering if raw sensor data
        if any(s in df.columns for s in SENSORS):
            df = engineer_features(df)

        cycles = df["cycle"].tolist() if "cycle" in df.columns else list(range(1, len(df) + 1))

        X = df[FEATURE_COLUMNS]
        X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLUMNS)

        predictions = np.clip(model.predict(X_scaled.values), 0, MAX_RUL)
        health_scores = np.clip(predictions / MAX_RUL * 100, 0, 100)

        latest = get_health(float(predictions[-1]))
        latest["rul"] = round(float(predictions[-1]), 1)

        return JSONResponse({
            "status": "success",
            "num_rows": len(predictions),
            "cycles": cycles,
            "predicted_rul": [round(float(p), 1) for p in predictions],
            "health_scores": [round(float(h), 1) for h in health_scores],
            "latest": latest,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
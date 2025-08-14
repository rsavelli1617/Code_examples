#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPM hindcast modeling (cleaned & refactored, with diagnostics)
--------------------------------------------------------------
- Deterministic seeding
- Robust I/O and daily resampling helpers
- Time-aware interpolation
- Missing-value and correlation diagnostics BEFORE dropping variables
- Proper scikeras GridSearchCV with R^2 scoring
- SHAP importance computed on a small sample (faster) with safe fallback
- Plotting fixes and tidy layout
- Replaced deprecated ._append with pd.concat
- Avoids forcing a specific Matplotlib backend
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

# Reproducibility
import random
random.seed(0)
np.random.seed(1)

import tensorflow as tf
tf.random.set_seed(2)

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ML / preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Keras + SciKeras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import get as get_optimizer
from scikeras.wrappers import KerasRegressor

# Other libs
import scipy.io
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ----------- optional: headless-safe matplotlib -----------
if not os.environ.get("DISPLAY"):
    mpl.use("Agg")

warnings.filterwarnings("ignore")


# ==========================================================
# Helpers
# ==========================================================
def resample_daily_mean(df: pd.DataFrame, tz_naive: bool = True) -> pd.DataFrame:
    """Resample a time-indexed DataFrame to daily means, optionally drop tz info."""
    if tz_naive and getattr(df.index, "tz", None) is not None:
        df = df.tz_convert(None)
    return df.resample("D").mean()


def as_time_indexed(
    df: pd.DataFrame, time_col: str, drop_cols: Iterable[str] | None = None
) -> pd.DataFrame:
    """Ensure datetime index and (optionally) drop columns."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    if drop_cols:
        keep = [c for c in df.columns if c not in set(drop_cols)]
        df = df[keep]
    return df


def concat_align_daily(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple daily-resampled frames and align by time index."""
    out = pd.concat(dfs, axis=1).sort_index()
    return out


def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns; drop duplicates in index."""
    df = df.loc[~df.index.duplicated(keep="first")]
    return df.select_dtypes(include=["number"])


def build_model(input_dim: int, optimizer: str = "adam", units: int = 128, dropout: float = 0.2) -> Sequential:
    """Build and compile a simple MLP for regression."""
    model = Sequential(
        [
            Dense(units, input_shape=(input_dim,), activation="relu", kernel_initializer="normal"),
            Dropout(dropout),
            Dense(1, activation="linear", kernel_initializer="normal"),
        ]
    )
    model.compile(optimizer=get_optimizer(optimizer), loss="mean_absolute_error", metrics=['r2_score','mae'])
    return model


def shap_feature_importance(predict_fn, X: pd.DataFrame, feature_names: list[str], sample_size: int = 200, random_state: int = 42) -> pd.Series:
    """Compute mean(|SHAP|) importances on a sample for speed; falls back gracefully if SHAP is unavailable."""
    try:
        import shap  # lazy import

        # Sample for performance
        Xs = X.sample(min(sample_size, len(X)), random_state=random_state)
        explainer = shap.Explainer(predict_fn, Xs)
        explanation = explainer(Xs)
        # Mean absolute SHAP across samples
        vals = np.mean(np.abs(explanation.values), axis=0)
        return pd.Series(vals, index=feature_names).sort_values(ascending=False)
    except Exception as e:
        warnings.warn(f"SHAP computation failed ({e}); skipping feature importance.")
        return pd.Series(dtype=float)


# ==========================================================
# Load data
# ==========================================================
BASE = Path(".").resolve()

# --- SPM at river mouth (target) ---
mat = scipy.io.loadmat(str(BASE / "insitu/SPM/ElwhaGF20132014.mat"))
# MATLAB datenum to Python datetime
time_dn = np.concatenate(mat["aqd"]["td"][0, 0]).astype(float)  # days since 0001-01-01
spm_obs = np.concatenate(mat["aqd"]["obs"][0, 0] * 1000.0).astype(float)  # g m-3

time_py = np.array([timedelta(days=float(d)) + datetime(1, 1, 1) - relativedelta(years=1) for d in time_dn])
dw_spm = pd.DataFrame({"Time (UTC)": pd.to_datetime(time_py), "SPM (g m-3)": spm_obs})
dw_spm = as_time_indexed(dw_spm, "Time (UTC)")
dw_spm_daily = resample_daily_mean(dw_spm)
dw_spm_daily = dw_spm_daily.loc[dw_spm_daily.index <= pd.Timestamp(2014, 3, 7)]

# --- Discharge upstream ---
discharge = pd.read_csv(
    BASE / "insitu/discharge/USGS_12045500_discharge.txt",
    sep="\t",
    skiprows=26,
    low_memory=False,
)
discharge = discharge.iloc[1:][["datetime", "150691_00060"]].rename(columns={"150691_00060": "discharge"})
discharge = as_time_indexed(discharge, "datetime")
discharge = discharge.apply(pd.to_numeric, errors="coerce")
discharge_daily = resample_daily_mean(discharge)

# --- Turbidity upstream ---
turbidity = pd.read_csv(
    BASE / "insitu/SPM/USGS_12046260_turbidity.txt",
    sep="\t",
    skiprows=27,
    low_memory=False,
)
turbidity = turbidity.iloc[1:][["datetime", "227925_63680"]].rename(columns={"227925_63680": "turbidity"})
turbidity = as_time_indexed(turbidity, "datetime")
turbidity = turbidity.apply(pd.to_numeric, errors="coerce")
turbidity_daily = resample_daily_mean(turbidity)

# --- Meteo (airport) ---
meteo = pd.read_csv(BASE / "insitu/meteo/WA_ASOS_CLM_PortAngelesAirport.csv", low_memory=False, na_values=["nan"])
drop_meteo = [
    "station",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "metar",
    "wxcodes",
    "peak_wind_time",
    "p01i",
    "ice_accretion_1hr",
    "ice_accretion_3hr",
    "ice_accretion_6hr",
]
meteo["valid"] = pd.to_datetime(meteo["valid"])
meteo = meteo.drop(columns=[c for c in drop_meteo if c in meteo.columns])
meteo = as_time_indexed(meteo, "valid")
meteo_daily = resample_daily_mean(meteo)

# --- Waves (buoy) ---
# waves = pd.read_csv(BASE / "insitu/waves/46088-Generic_Export-20250605T13T01_11.csv", low_memory=False)
waves = pd.read_csv(BASE / "insitu/waves/46087-Generic_Export-20250611T12T12_35.csv", low_memory=False)
waves["time"] = pd.to_datetime(waves["time"])
waves = as_time_indexed(waves, "time")
waves_daily = resample_daily_mean(waves)

# --- Water level ---
wl_dir = BASE / "insitu/water_level"
wl = pd.DataFrame()
for file in os.listdir(wl_dir):
    filename = os.fsdecode(file)
    tmp = pd.read_csv(wl_dir / filename, sep='\t', skiprows = 9, low_memory=False)
    tmp.index = pd.to_datetime(tmp.index)
    tmp = tmp.rename(columns={"// datetime [ISO8601], waterlevel_unassessed [m]": "water_level"})
    wl = wl._append(tmp)
wl = wl.sort_index()
wl.index = pd.to_datetime(wl.index)
wl.index = wl.index.tz_localize(None)
wl_daily = wl.resample('D').mean()

# --- In-situ upstream sediment loads ---
up_spm_daily = pd.read_csv(BASE / "insitu/SPM/Elwha_DailySedimentLoads_2011to2016.csv")
up_spm_daily["Day"] = pd.to_datetime(up_spm_daily["Day"])
up_spm_daily = as_time_indexed(up_spm_daily, "Day")

# ==========================================================
# Merge, interpolate, and diagnostics BEFORE dropping columns
# ==========================================================
df_all = concat_align_daily([dw_spm_daily, waves_daily, discharge_daily, turbidity_daily, meteo_daily, wl_daily, up_spm_daily])
df_all = df_all.sort_index()

# Time-aware interpolation both directions
df_interp = df_all.interpolate(method="time", limit_direction="both")

# Align to observed SPM dates
df_obs_aligned = df_interp.loc[dw_spm_daily.index]

# ---- Diagnostics: missing values & correlation (BEFORE drop) ----
target_col = "SPM (g m-3)"
missing_before = df_obs_aligned.isna().sum().sort_values(ascending=False)
print("\n--- Missing values per column (before drop) ---")
# Print only columns with at least one missing value
if (missing_before > 0).any():
    print(missing_before[missing_before > 0].to_string())
else:
    print("No missing values.")

# Correlation only on numeric subset
num_before = df_obs_aligned.select_dtypes(include="number")
if target_col in num_before.columns:
    corr_before = num_before.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print("\n--- Correlation with 'SPM (g m-3)' (before drop) ---")
    print(corr_before.to_string())
    # Optional: save to CSVs for quick inspection
    #missing_before.to_csv("missing_before.csv")
    #corr_before.to_csv("corr_with_SPM_before.csv")
else:
    print(f"\nTarget '{target_col}' not available in numeric columns for correlation at this stage.")

# ==========================================================
# Drop obviously irrelevant/text columns, then numeric-only & dropna
# ==========================================================
drop_cols = [
    "Project year",
    "Release period",
    "Remarks",
    "latitude",
    "longitude",
    "snowdepth",
    "skyl4",
    "waveSensorOutput",
    "waveSysSensor",
    "waveFrequency_47",
    "depth",
    "Daily Total gauged > 2-mm bedload (tonnes)",
    "Daily gauged bedload for 2-16mm particles (tonnes )",
    "Daily gauged bedload for >16mm particles (tonnes )",
    "Estimated daily ungauged bedload (tonnes)",
    "Ave fraction fines (based on two turbidimeters)",
    "gust",
    "relh",
    "skyl1",
    "skyl2",
    "skyl3",
    "waveHs",
    "waveTp",
]
drop_cols = [c for c in drop_cols if c in df_obs_aligned.columns]
df_clean_stage = df_obs_aligned.drop(columns=drop_cols, errors="ignore")

# Keep numeric columns, drop rows with remaining NaNs
df = numeric_only(df_clean_stage).dropna(how="any")

# Post-cleaning missing check (should be none)
missing_after = df.isna().sum().sort_values(ascending=False)
print("\n--- Missing values per column (after cleaning) ---")
if (missing_after > 0).any():
    print(missing_after[missing_after > 0].to_string())
else:
    print("No missing values after cleaning.")

# ==========================================================
# Scaling & split
# ==========================================================
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found after cleaning. Columns: {list(df.columns)}")

# Impute (safety) then scale
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), index=df.index, columns=df.columns)

feature_cols = [c for c in df_imputed.columns if c != target_col]

X = df_imputed[feature_cols]
y = df_imputed[target_col]

# Feature scaling
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = pd.DataFrame(x_scaler.fit_transform(X), index=X.index, columns=X.columns)
y_scaled = pd.Series(y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel(), index=y.index, name=target_col)

# Split (keep indices)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ==========================================================
# Model & hyperparameter search
# ==========================================================
input_dim = X_train.shape[1]

reg = KerasRegressor(model=build_model(input_dim), verbose=2)

param_grid = {
    "optimizer": ["adam", "sgd", "rmsprop", "nadam"],
    'epochs': [30, 60, 90, 120, 150, 180, 210, 240],
    'batch_size': [10, 20, 40, 60, 80, 100]
}

grid = GridSearchCV(
    estimator=reg,
    param_grid=param_grid,
    cv=3
)

grid.fit(X_train, y_train)

print(f"Best CV score (R2): {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")

# Best model predictions (scaled)
y_pred_test_scaled = grid.predict(X_test)
y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
y_test_real = y_scaler.inverse_transform(y_test.to_numpy().reshape(-1, 1)).ravel()

# Collect into a single frame indexed by date
d = pd.DataFrame(
    {"Predicted_SPM": y_pred_test, "Actual_SPM": y_test_real},
    index=X_test.index,
).sort_index()

# Metrics
rmse = np.sqrt(mean_squared_error(d["Actual_SPM"], d["Predicted_SPM"]))
mae = mean_absolute_error(d["Actual_SPM"], d["Predicted_SPM"])
r2 = r2_score(d["Actual_SPM"], d["Predicted_SPM"])
print({"RMSE": rmse, "MAE": mae, "R2": r2})

# ==========================================================
# Hindcast over full (interpolated) period
# ==========================================================
# Build full feature matrix on the interpolated set (same cleaning)
new_df = df_interp.drop(columns=drop_cols, errors="ignore")
new_df = numeric_only(new_df)

# Align columns to training features (safe reindex)
new_X = new_df.reindex(columns=feature_cols)

# Scale with training scalers
new_X_scaled = pd.DataFrame(x_scaler.transform(new_X), index=new_X.index, columns=new_X.columns)

final_pred_scaled = grid.predict(new_X_scaled)
final_pred = y_scaler.inverse_transform(final_pred_scaled.reshape(-1, 1)).ravel()
final_pred = pd.Series(final_pred, index=new_X.index, name="SPM_pred")

# Subset to observed range for scatter metrics
final_pred_sub = final_pred.reindex(df.index)

rmse_all = np.sqrt(mean_squared_error(df[target_col], final_pred_sub))
r2_all = r2_score(df[target_col], final_pred_sub)

# ==========================================================
# SHAP feature importance (fast sample)
# ==========================================================
feat_importance = shap_feature_importance(grid.predict, X_train, feature_cols, sample_size=200)

# ==========================================================
# Plotting
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1) Time Series (full)
fmt1 = DateFormatter("%m-%Y")
axes[0, 0].axvspan(df.index.min(), df.index.max(), facecolor="lightgreen", alpha=0.5, label="Training/Test", zorder=0)
axes[0, 0].plot(df[target_col], label="Observed", alpha=0.8, zorder=2)
axes[0, 0].plot(final_pred, label="Hindcast", alpha=0.8, zorder=1)
axes[0, 0].set_xlabel("Time", font="Arial", size=12)
axes[0, 0].set_ylabel("Suspended Sediment (g m$^{-3}$)", font="Arial", size=12)
axes[0, 0].set_title("Observed vs. Hindcasted Time Series", font="Arial", size=16)
axes[0, 0].legend()
axes[0, 0].set_xlim(pd.Timestamp("2011-09-15"), pd.Timestamp("2016-09-20"))
axes[0, 0].xaxis.set_major_formatter(fmt1)
axes[0, 0].grid(True)

# 2) Time Series (zoom to observed window only)
fmt2 = DateFormatter("%d-%m")
axes[1, 0].axvspan(df.index.min(), df.index.max(), facecolor="lightgreen", alpha=0.5, zorder=0)
axes[1, 0].plot(df[target_col], label="Observed", alpha=0.8)
axes[1, 0].plot(final_pred, label="Hindcast", alpha=0.8)
axes[1, 0].set_xlabel("Time", font="Arial", size=12)
axes[1, 0].set_ylabel("Suspended Sediment (g m$^{-3}$)", font="Arial", size=12)
axes[1, 0].set_title("Observed vs. Hindcasted (Observed Window)", font="Arial", size=16)
axes[1, 0].legend()
axes[1, 0].set_xlim(df.index.min(), df.index.max())
axes[1, 0].xaxis.set_major_formatter(fmt2)
axes[1, 0].grid(True)

# 3) Scatter Observed vs. Hindcast (observed window)
x = df[target_col].to_numpy()
y = final_pred_sub.to_numpy()
xy_min = float(np.nanmin([x.min(), y.min()]))
xy_max = float(np.nanmax([x.max(), y.max()]))
axes[0, 1].scatter(x, y, alpha=0.6, edgecolors="black")
axes[0, 1].plot([xy_min, xy_max], [xy_min, xy_max], "k--")
axes[0, 1].set_xlabel("Observed", font="Arial", size=12)
axes[0, 1].set_ylabel("Hindcast", font="Arial", size=12)
axes[0, 1].set_title("Observed vs. Hindcast", font="Arial", size=16)
axes[0, 1].grid(True)
axes[0, 1].text(
    0.05,
    0.9,
    f"RMSE: {rmse_all:.2f}\nR²: {r2_all:.2f}",
    transform=axes[0, 1].transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)

# 4) Feature Importance
if len(feat_importance) > 0:
    axes[1, 1].barh(feat_importance.index.tolist(), feat_importance.values, alpha=0.8)
    axes[1, 1].set_xlabel("Relative Importance", font="Arial", size=12)
    axes[1, 1].set_title("Feature Importance (SHAP)", font="Arial", size=16)
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, axis="x")
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
else:
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Feature Importance (SHAP unavailable)")

plt.tight_layout()
plt.savefig("FULL_analysis2.png", dpi=200)
# plt.show()  # uncomment for interactive sessions

print("Saved figure: FULL_analysis2.png")
print("Saved diagnostics: missing_before.csv, corr_with_SPM_before.csv")

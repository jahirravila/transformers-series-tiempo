"""Utilidades de series de tiempo: carga, partición cronológica, escalado, ventanas, líneas base y métricas."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
LOOKBACK = 60          # días de contexto que ve el modelo
HORIZON = 7            # días a pronosticar
SPLITS = {"train": ("1981-01-01", "1986-12-31"), "val": ("1987-01-01", "1988-12-31"), "test": ("1989-01-01", "1990-12-31")}

for d in ("data", "figures", "results", "modelos"):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def load_series(path: str = "data/daily-min-temperatures.csv") -> pd.Series:
    """Temperatura mínima diaria de Melbourne (°C), con frecuencia diaria completa."""
    if not os.path.exists(path):
        pd.read_csv(URL).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["Date"])
    s = pd.to_numeric(df["Temp"], errors="coerce")
    s.index = df["Date"]
    full = s.asfreq("D")                       # inserta las fechas faltantes como NaN
    return full


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Codificación cíclica del día del año: (sin, cos). Evita el salto artificial 31-dic → 1-ene."""
    doy = index.dayofyear.values
    ang = 2 * np.pi * doy / 365.25
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype("float32")


class Scaler:
    """Estandarización z-score ajustada SÓLO con el conjunto de entrenamiento."""
    def fit(self, x):
        self.mu, self.sigma = float(np.mean(x)), float(np.std(x)); return self
    def transform(self, x): return (np.asarray(x) - self.mu) / self.sigma
    def inverse(self, z): return np.asarray(z) * self.sigma + self.mu


def make_windows(values_z, feats, dates, start, end, lookback=LOOKBACK, horizon=HORIZON):
    """Ventanas deslizantes: X = [temp_z, sin, cos] de los `lookback` días previos,
    y = temp_z de los `horizon` días siguientes. Sólo se aceptan ventanas cuyo PRIMER día
    a pronosticar cae en [start, end] y cuyo último día objetivo no supera `end`
    (el contexto sí puede venir del periodo anterior, que ya es pasado conocido)."""
    X_in = np.concatenate([values_z[:, None], feats], axis=1).astype("float32")
    t0 = np.searchsorted(dates, np.datetime64(start)); t1 = np.searchsorted(dates, np.datetime64(end), side="right")
    X, Y, D, F = [], [], [], []
    for t in range(max(t0, lookback), t1 - horizon + 1):
        X.append(X_in[t - lookback:t]); Y.append(values_z[t:t + horizon])
        F.append(feats[t:t + horizon]); D.append(dates[t])
    return np.array(X), np.array(Y, dtype="float32"), np.array(F, dtype="float32"), np.array(D)


def prepare(lookback=LOOKBACK, horizon=HORIZON):
    """Pipeline completo: serie limpia, escalador y ventanas por partición."""
    s = load_series()
    n_missing = int(s.isna().sum())
    s = s.interpolate(limit_direction="both")
    dates = s.index.values
    tr_mask = (s.index >= SPLITS["train"][0]) & (s.index <= SPLITS["train"][1])
    scaler = Scaler().fit(s.values[tr_mask])
    z = scaler.transform(s.values).astype("float32")
    feats = calendar_features(s.index)
    data = {"series": s, "scaler": scaler, "n_missing": n_missing, "z": z, "feats": feats}
    for name, (a, b) in SPLITS.items():
        X, Y, F, D = make_windows(z, feats, dates, a, b, lookback, horizon)
        data[name] = {"X": X, "Y": Y, "F": F, "dates": D}
    return data


# ----------------------------------------------------------------------------- líneas base
def persistence(X):
    """Pronóstico ingenuo: los próximos días serán iguales al último observado."""
    return np.repeat(X[:, -1:, 0], HORIZON, axis=1)


def climatology_model(series: pd.Series, scaler: Scaler, window: int = 15):
    """Media por día del año en train (suavizada con media móvil circular). Devuelve f(fechas)->z."""
    tr = series[SPLITS["train"][0]:SPLITS["train"][1]]
    clim = tr.groupby(tr.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
    ext = pd.concat([clim.iloc[-window:], clim, clim.iloc[:window]])
    smooth = ext.rolling(window, center=True).mean().iloc[window:-window]
    def predict(first_dates):
        out = []
        for d0 in pd.to_datetime(first_dates):
            days = pd.date_range(d0, periods=HORIZON, freq="D").dayofyear
            out.append(scaler.transform(smooth.loc[days].values))
        return np.array(out, dtype="float32")
    return predict, smooth


# ----------------------------------------------------------------------------- métricas
def metrics(y_true_z, y_pred_z, scaler: Scaler) -> dict:
    """MAE y RMSE en °C (global y por horizonte)."""
    yt, yp = scaler.inverse(y_true_z), scaler.inverse(y_pred_z)
    err = yp - yt
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAE_h": np.abs(err).mean(0).tolist(), "MAE_d1": float(np.abs(err[:, 0]).mean()),
            "MAE_d7": float(np.abs(err[:, -1]).mean())}


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=conv)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

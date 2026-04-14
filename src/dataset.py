"""
dataset.py
==========
Synthetic PSG dataset generator for Sleep Apnea screening.
"""

from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def _patient_cohort(n: int, severity: str) -> pd.DataFrame:
    configs = {
        #           ahi_lo  hi   spo2m_mu sd  spo2n_mu sd   fl_mu  sd    ef_mu  sd    bmi_mu sd
        "None":     (0,  5,  97.0, 0.5, 95.0, 1.0, 0.88, 0.06, 0.20, 0.05, 24.0, 3.0),
        "Mild":     (5,  15, 95.5, 0.8, 91.0, 2.0, 0.72, 0.08, 0.40, 0.08, 28.5, 3.5),
        "Moderate": (15, 30, 94.0, 1.0, 87.0, 3.0, 0.55, 0.10, 0.60, 0.09, 32.0, 4.0),
        "Severe":   (30, 80, 91.5, 1.5, 81.0, 4.0, 0.38, 0.12, 0.78, 0.10, 36.5, 5.0),
    }
    (ahi_lo, ahi_hi, spo2m_mu, spo2m_sd, spo2n_mu, spo2n_sd,
     fl_mu, fl_sd, ef_mu, ef_sd, bmi_mu, bmi_sd) = configs[severity]

    n_pts = n
    ages  = np.random.randint(30, 75, n_pts)
    sexes = np.random.choice(["M", "F"], n_pts, p=[0.60, 0.40])
    bmis  = np.clip(np.random.normal(bmi_mu, bmi_sd, n_pts), 16, 55)
    ahis  = np.random.uniform(ahi_lo, ahi_hi, n_pts).round(1)
    spo2m = np.clip(np.random.normal(spo2m_mu, spo2m_sd, n_pts), 85, 99).round(1)
    spo2n = np.clip(np.random.normal(spo2n_mu, spo2n_sd, n_pts), 60, 98).round(1)
    flows = np.clip(np.random.normal(fl_mu, fl_sd, n_pts), 0.05, 1.0).round(3)
    effs  = np.clip(np.random.normal(ef_mu, ef_sd, n_pts), 0.05, 1.0).round(3)
    ids   = [f"PT{severity[0]}{str(i+1).zfill(3)}" for i in range(n_pts)]

    return pd.DataFrame({
        "patient_id":           ids,
        "age":                  ages,
        "sex":                  sexes,
        "bmi":                  bmis.round(1),
        "spo2_mean":            spo2m,
        "spo2_min":             spo2n,
        "nasal_airflow_mean":   flows,
        "thoracic_effort_mean": effs,
        "ahi":                  ahis,
        "severity_label":       severity,
    })


def generate_dataset(
    n_none=30, n_mild=30, n_moderate=25, n_severe=15,
    save_path="data/sleep_apnea_dataset.csv"
) -> pd.DataFrame:
    df = pd.concat([
        _patient_cohort(n_none,     "None"),
        _patient_cohort(n_mild,     "Mild"),
        _patient_cohort(n_moderate, "Moderate"),
        _patient_cohort(n_severe,   "Severe"),
    ], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Dataset] Saved {len(df)} records -> {save_path}")
    return df

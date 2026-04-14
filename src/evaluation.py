"""
evaluation.py
=============
Evaluates system performance against ground-truth severity labels.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import OrchestratorAgent
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(csv_path="data/sleep_apnea_dataset.csv", max_samples=50) -> dict:
    df           = pd.read_csv(csv_path).head(max_samples)
    orchestrator = OrchestratorAgent()
    y_true, y_pred = [], []
    failed = 0

    print(f"\nEvaluating on {len(df)} patients (ground-truth AHI hidden)...")

    for _, row in df.iterrows():
        raw        = row.to_dict()
        true_label = raw.pop("severity_label")
        raw.pop("ahi", None)              # hide ground truth

        try:
            report     = orchestrator.run(raw)
            pred_label = report.get("severity", "Unknown")
        except Exception:
            pred_label = "Unknown"
            failed    += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

    labels = ["None", "Mild", "Moderate", "Severe"]
    acc    = accuracy_score(y_true, y_pred)

    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  Samples   : {len(df)}")
    print(f"  Failed    : {failed}")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Confusion Matrix (rows=True, cols=Predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print("="*60)

    return {"accuracy": round(acc, 4), "samples": len(df), "failed": failed}

# src/drift.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def main():
    reference = pd.read_csv("reference_val_predictions.csv")
    current = pd.read_csv("val_predictions.csv")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report.save_html("drift_report.html")

    print("Drift report generated.")


if __name__ == "__main__":
    main()

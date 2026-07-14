from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FILE = Path("Baselines_with_subject") / "sub-s01_ses-1_run-001_bl-01.csv"
REPORT_FILE = Path("single_file_loading_report.txt")

IGNORED_COLUMNS = {"stream_time", "Trigger", "Sample Counter", "EOG"}

EEG_CHANNELS = {
    "Fp1",
    "Fpz",
    "Fp2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFz",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POz",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "Oz",
    "O2",
    "Iz",
}


def is_ignored_column(column_name: str) -> bool:
    return column_name in IGNORED_COLUMNS or column_name.startswith("Bipolar_")


def format_numbered_list(values: list[str]) -> list[str]:
    if not values:
        return ["  (none)"]
    return [f"  {index:02d}. {value}" for index, value in enumerate(values, start=1)]


def describe_channel_audit(df: pd.DataFrame, eeg_channel_names: list[str]) -> list[str]:
    all_columns = list(df.columns)
    ignored_columns = [column for column in all_columns if is_ignored_column(column)]
    unselected_non_ignored_columns = [
        column
        for column in all_columns
        if column not in eeg_channel_names and column not in ignored_columns
    ]
    recognized_eeg_labels_absent_from_csv = [
        channel for channel in sorted(EEG_CHANNELS) if channel not in all_columns
    ]

    non_bipolar_signal_columns = [
        column
        for column in all_columns
        if column not in {"stream_time", "Trigger", "Sample Counter"}
        and not column.startswith("Bipolar_")
    ]

    if unselected_non_ignored_columns:
        conclusion = (
            "One or more CSV columns were neither selected as EEG nor ignored. "
            "Check these as possible renamed EEG channels."
        )
    elif "EOG" in ignored_columns and len(non_bipolar_signal_columns) == len(eeg_channel_names) + 1:
        conclusion = (
            "The CSV has 64 non-bipolar signal columns: 63 selected EEG channels plus "
            "EOG, which is intentionally ignored. No EEG-labelled CSV column appears "
            "to be missing, renamed, or incorrectly ignored by the current rules."
        )
    else:
        conclusion = (
            "All non-ignored EEG-labelled CSV columns were selected. No extra "
            "unselected CSV columns were found."
        )

    report_lines = [
        "",
        "Column audit",
        f"Total CSV columns: {len(all_columns)}",
        f"Selected EEG columns: {len(eeg_channel_names)}",
        f"Ignored columns: {len(ignored_columns)}",
        f"Non-bipolar signal columns before ignoring EOG: {len(non_bipolar_signal_columns)}",
        "",
        "All CSV columns:",
        *format_numbered_list(all_columns),
        "",
        "Selected EEG columns:",
        *format_numbered_list(eeg_channel_names),
        "",
        "Ignored columns:",
        *format_numbered_list(ignored_columns),
        "",
        "CSV columns not selected as EEG and not ignored:",
        *format_numbered_list(unselected_non_ignored_columns),
        "",
        "Other EEG labels recognized by the script but absent from this CSV:",
        *format_numbered_list(recognized_eeg_labels_absent_from_csv),
        "",
        f"Conclusion: {conclusion}",
    ]
    return report_lines


def infer_sampling_frequency(stream_time: pd.Series) -> float:
    times = pd.to_numeric(stream_time, errors="coerce").dropna().to_numpy(dtype=float)
    if times.size < 2:
        raise ValueError("stream_time must contain at least two numeric samples.")

    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        raise ValueError("stream_time does not contain positive sample-to-sample differences.")

    median_diff = float(np.median(positive_diffs))
    return 1.0 / median_diff


def build_report(file_path: Path) -> str:
    df = pd.read_csv(file_path, sep=";")
    df.columns = [str(column).strip() for column in df.columns]

    if "stream_time" not in df.columns:
        raise ValueError("Missing required column: stream_time")

    eeg_channel_names = [
        column
        for column in df.columns
        if column in EEG_CHANNELS and not is_ignored_column(column)
    ]
    if not eeg_channel_names:
        raise ValueError("No EEG channels were identified in the CSV header.")

    eeg_data_mV = df[eeg_channel_names].apply(pd.to_numeric, errors="coerce")
    eeg_data_volts = eeg_data_mV * 1e-3
    eeg_data_uV = eeg_data_volts * 1e6

    sample_count = len(df)
    stream_time = pd.to_numeric(df["stream_time"], errors="coerce")
    duration_seconds = float(stream_time.dropna().iloc[-1] - stream_time.dropna().iloc[0])
    fs = infer_sampling_frequency(stream_time)

    values_uV = eeg_data_uV.to_numpy(dtype=float).ravel()
    finite_values_uV = values_uV[np.isfinite(values_uV)]
    if finite_values_uV.size == 0:
        raise ValueError("EEG channels do not contain finite numeric values.")

    max_abs_uV = float(np.max(np.abs(finite_values_uV)))
    plausibility = (
        "OK: EEG values are within a broad expected microvolt range."
        if max_abs_uV <= 1000.0
        else "WARNING: EEG values exceed +/-1000 uV; inspect units or artifacts."
    )

    report_lines = [
        "Single-file CSV loading report",
        f"Input file: {file_path}",
        "",
        f"Number of samples: {sample_count}",
        f"Duration in seconds: {duration_seconds:.6f}",
        f"Inferred sampling frequency: {fs:.6f} Hz",
        f"Number of EEG channels: {len(eeg_channel_names)}",
        f"Channel names used: {', '.join(eeg_channel_names)}",
        f"Minimum EEG value: {float(np.min(finite_values_uV)):.6f} uV",
        f"Maximum EEG value: {float(np.max(finite_values_uV)):.6f} uV",
        f"Mean EEG value: {float(np.mean(finite_values_uV)):.6f} uV",
        f"Standard deviation: {float(np.std(finite_values_uV)):.6f} uV",
        f"Plausibility check: {plausibility}",
        *describe_channel_audit(df, eeg_channel_names),
    ]
    return "\n".join(report_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load one baseline CSV and validate EEG values without preprocessing."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=DEFAULT_FILE,
        type=Path,
        help=f"Baseline CSV file to load. Defaults to {DEFAULT_FILE}",
    )
    parser.add_argument(
        "--report",
        default=REPORT_FILE,
        type=Path,
        help=f"Report output path. Defaults to {REPORT_FILE}",
    )
    args = parser.parse_args()

    report = build_report(args.file_path)
    print(report, end="")
    args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

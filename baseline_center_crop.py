from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_FILE = Path("Baselines_with_subject") / "sub-s01_ses-1_run-001_bl-01.csv"
REPORT_FILE = Path("baseline_cropping_report.txt")
TARGET_DURATION_SECONDS = 60.0


@dataclass(frozen=True)
class CropResult:
    original_duration: float
    crop_start: float | None
    crop_end: float | None
    cropped_duration: float
    cropped_samples: int
    status: str


def crop_centered_60_seconds(df: pd.DataFrame) -> tuple[pd.DataFrame | None, CropResult]:
    if "stream_time" not in df.columns:
        raise ValueError("Missing required column: stream_time")

    stream_time = pd.to_numeric(df["stream_time"], errors="coerce")
    if stream_time.isna().any():
        raise ValueError("stream_time contains non-numeric or missing values")

    first_stream_time = float(stream_time.iloc[0])
    last_stream_time = float(stream_time.iloc[-1])
    original_duration = last_stream_time - first_stream_time

    if original_duration < TARGET_DURATION_SECONDS:
        return None, CropResult(
            original_duration=original_duration,
            crop_start=None,
            crop_end=None,
            cropped_duration=0.0,
            cropped_samples=0,
            status="insufficient_duration",
        )

    crop_start = first_stream_time + (original_duration - TARGET_DURATION_SECONDS) / 2.0
    crop_end = crop_start + TARGET_DURATION_SECONDS
    crop_mask = (stream_time >= crop_start) & (stream_time <= crop_end)
    cropped_df = df.loc[crop_mask].copy()

    return cropped_df, CropResult(
        original_duration=original_duration,
        crop_start=crop_start,
        crop_end=crop_end,
        cropped_duration=crop_end - crop_start,
        cropped_samples=len(cropped_df),
        status="OK",
    )


def format_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def build_report(file_path: Path) -> str:
    df = pd.read_csv(file_path, sep=";")
    _, result = crop_centered_60_seconds(df)

    report_lines = [
        "Baseline centered 60-second cropping report",
        f"Input file: {file_path}",
        "",
        f"Original duration: {format_seconds(result.original_duration)} seconds",
        f"Crop start: {format_seconds(result.crop_start)}",
        f"Crop end: {format_seconds(result.crop_end)}",
        f"Cropped duration: {format_seconds(result.cropped_duration)} seconds",
        f"Number of cropped samples: {result.cropped_samples}",
        f"Status: {result.status}",
    ]
    return "\n".join(report_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply centered 60-second cropping to one baseline CSV recording."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=DEFAULT_FILE,
        type=Path,
        help=f"Baseline CSV file to crop. Defaults to {DEFAULT_FILE}",
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

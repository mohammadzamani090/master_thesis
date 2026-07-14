from __future__ import annotations

"""Continuous baseline EEG preprocessing pipeline.

The file is organized in the same order as the processing workflow:
loading_data -> validation -> channel_separation -> MNE object creation ->
detrending -> bad-channel detection/interpolation -> ZapLine/notch cleanup ->
Butterworth filtering -> 60-second crop -> EEG average reference -> epoching ->
bad-epoch rejection -> output and QC reporting.
"""

import argparse
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("baseline_preprocessing")


# errors: explicit failure types keep batch-mode reports clear and searchable.
class PreprocessingError(RuntimeError):
    """Raised when a recording cannot be processed safely."""


class InsufficientDurationError(PreprocessingError):
    """Raised when a recording is shorter than the requested crop duration."""


# configuration: all tunable pipeline parameters live here.
@dataclass
class PreprocessingConfig:
    """Configuration for continuous baseline EEG preprocessing."""

    csv_separator: str = ";"
    input_file_extensions: tuple[str, ...] = (".csv",)
    timestamp_column: str = "stream_time"
    ignored_columns: tuple[str, ...] = ("Trigger", "Sample Counter")
    ignored_prefixes: tuple[str, ...] = ("Bipolar_",)
    amplitude_unit: str = "mV"
    input_orientation: str = "auto"

    nominal_fs: float = 1024.0
    fs_warning_hz: float = 0.5
    sampling_frequency_mode: str = "nominal"

    eog_name_token: str = "EOG"
    eog_required: bool = False
    montage: str | None = "standard_1020"
    montage_on_missing: str = "warn"

    detrend_type: str = "constant"
    enable_linear_detrend: bool = False

    low_variance_threshold: float = 1e-18
    zero_variance_threshold: float = 0.0
    flat_diff_tolerance: float = 1e-12
    max_constant_segment_seconds: float = 1.0
    max_interp_fraction_soft: float = 0.15
    max_interp_fraction_hard: float = 0.20

    zapline_enabled: bool = True
    zapline_eog_enabled: bool = False
    line_noise_hz: float = 50.0
    zap_harmonics: tuple[int, ...] = (1, 2, 3)
    zap_nremove_eeg: int = 5
    zap_nremove_eog: int = 1
    fallback_notch_enabled: bool = True
    notch_quality_factor: float = 30.0

    butterworth_order: int = 4
    eeg_low_cut_hz: float = 0.5
    eeg_high_cut_hz: float = 80.0
    eog_lowpass_hz: float = 10.0
    filter_padlen: int | None = 150

    crop_duration_seconds: float = 60.0
    crop_start_time: float | None = None
    crop_start_offset_seconds: float | None = None
    crop_duration_tolerance_seconds: float = 0.01
    crop_sample_count_tolerance: int = 1
    crop_enforce_nominal_sample_count: bool = True

    epoch_duration_seconds: float = 4.0
    epoch_overlap_seconds: float = 0.0

    epoch_peak_to_peak_threshold_volts: float = 350e-6
    epoch_large_excursion_threshold_volts: float = 350e-6
    epoch_extreme_peak_to_peak_threshold_volts: float = 1000e-6
    epoch_flat_variance_threshold: float = 1e-18
    epoch_flat_peak_to_peak_threshold_volts: float = 1e-12
    max_bad_channels: int | float = 5

    generate_plots: bool = False
    plot_channel_count: int = 6
    max_rejected_epoch_plots: int = 10
    rejected_epoch_plot_channel_count: int = 8


# data_containers: lightweight structures passed between pipeline stages.
@dataclass
class BaselineRecording:
    """Loaded continuous baseline data in channel-by-sample orientation."""

    data: np.ndarray
    timestamps: np.ndarray
    ch_names: list[str]
    source_path: Path
    original_duration: float
    effective_fs: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelGroups:
    """EEG and EOG channel split."""

    eeg_indices: list[int]
    eog_indices: list[int]
    eeg_ch_names: list[str]
    eog_ch_names: list[str]


@dataclass
class ValidationReport:
    """Basic validation metadata and warnings."""

    n_channels: int
    n_samples: int
    duration: float
    effective_fs: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class CropResult:
    """Metadata for the retained analysis interval."""

    data: np.ndarray
    timestamps: np.ndarray
    crop_start: float
    crop_end: float
    requested_duration: float
    actual_timestamp_span: float
    n_samples: int
    expected_samples: int
    sample_coverage_duration: float
    first_to_last_timestamp_span: float


@dataclass
class EpochResult:
    """Fixed-length EEG and EOG epochs."""

    eeg_epochs: np.ndarray
    eog_epochs: np.ndarray | None
    epoch_start_times: np.ndarray
    epoch_end_times: np.ndarray
    samples_per_epoch: int


@dataclass
class RejectionResult:
    """Epoch rejection output."""

    retained_indices: list[int]
    rejected_indices: list[int]
    rejection_reasons: dict[int, list[str]]
    epoch_quality_details: dict[int, dict[str, Any]]


@dataclass
class ProcessingSummary:
    """Concise result for one processed file."""

    input_path: Path
    output_path: Path | None
    qc_report_path: Path | None
    original_duration: float
    cropped_duration: float
    n_eeg_channels: int
    n_eog_channels: int
    interpolated_channels: list[str]
    total_epochs: int
    rejected_epochs: int
    retained_epochs: int
    quality_status: str


# configuration_loading: optional JSON config overrides the dataclass defaults.
def validate_config(config: PreprocessingConfig) -> None:
    """Validate configuration values that affect file discovery and sample timing."""
    mode = config.sampling_frequency_mode.strip().lower()
    if mode not in {"nominal", "effective"}:
        raise ValueError(
            "sampling_frequency_mode must be either 'nominal' or 'effective', "
            f"got {config.sampling_frequency_mode!r}."
        )
    if config.nominal_fs <= 0:
        raise ValueError("nominal_fs must be positive.")
    if config.fs_warning_hz < 0:
        raise ValueError("fs_warning_hz cannot be negative.")
    if config.crop_sample_count_tolerance < 0:
        raise ValueError("crop_sample_count_tolerance cannot be negative.")
    if config.max_rejected_epoch_plots < 0:
        raise ValueError("max_rejected_epoch_plots cannot be negative.")
    if config.rejected_epoch_plot_channel_count < 1:
        raise ValueError("rejected_epoch_plot_channel_count must be at least 1.")

    extensions = tuple(str(ext).strip().lower() for ext in config.input_file_extensions)
    if not extensions or any(not ext.startswith(".") for ext in extensions):
        raise ValueError(
            "input_file_extensions must contain one or more suffixes beginning with '.', "
            "for example ('.csv',)."
        )


def load_config(config_path: Path | None) -> PreprocessingConfig:
    """Load optional JSON configuration over the dataclass defaults."""
    config = PreprocessingConfig()
    if config_path is None:
        validate_config(config)
        return config

    with config_path.open("r", encoding="utf-8") as handle:
        overrides = json.load(handle)

    valid_fields = set(asdict(config).keys())
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown configuration field(s): {', '.join(unknown)}")

    for key, value in overrides.items():
        current = getattr(config, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(config, key, value)
    validate_config(config)
    return config


def determine_processing_fs(
    recording: BaselineRecording,
    config: PreprocessingConfig,
) -> float:
    """Choose the single sampling frequency used by all processing stages."""
    validate_config(config)
    mode = config.sampling_frequency_mode.strip().lower()
    processing_fs = config.nominal_fs if mode == "nominal" else recording.effective_fs
    if not np.isfinite(processing_fs) or processing_fs <= 0:
        raise PreprocessingError(
            f"Selected processing sampling frequency is invalid: {processing_fs!r}."
        )
    return float(processing_fs)


# optional_dependencies: full preprocessing imports SciPy, MNE, meegkit and matplotlib lazily.
def require_module(module_name: str) -> Any:
    """Import an optional processing dependency with a clear error message."""
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise PreprocessingError(
            f"Missing required dependency '{module_name}'. Install it in the EEG "
            "processing environment to run the full pipeline."
        ) from exc


# unit_conversion: convert input mV/uV/V amplitudes to volts for MNE and thresholds.
def amplitude_scale_to_volts(unit: str) -> float:
    """Return a scale factor that converts the configured amplitude unit to volts."""
    normalized = unit.strip().lower()
    scales = {
        "v": 1.0,
        "volt": 1.0,
        "volts": 1.0,
        "mv": 1e-3,
        "millivolt": 1e-3,
        "millivolts": 1e-3,
        "uv": 1e-6,
        "microvolt": 1e-6,
        "microvolts": 1e-6,
    }
    if normalized not in scales:
        raise ValueError(f"Unsupported amplitude unit: {unit}")
    return scales[normalized]


# column_filtering: remove metadata and auxiliary channels before EEG/EOG separation.
def is_ignored_column(column: str, config: PreprocessingConfig) -> bool:
    """Return True for metadata or auxiliary columns that are not EEG/EOG."""
    return column in config.ignored_columns or any(
        column.startswith(prefix) for prefix in config.ignored_prefixes
    )


# sampling_rate: infer effective fs from stream_time rather than trusting metadata.
def infer_sampling_frequency(timestamps: np.ndarray) -> float:
    """Infer sampling frequency from the median positive timestamp difference."""
    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        raise ValueError("Cannot infer sampling frequency from non-increasing timestamps.")
    return float(1.0 / np.median(positive_diffs))


# shape_handling: accept either samples x channels or channels x samples.
def standardize_orientation(
    data: np.ndarray,
    timestamps: np.ndarray,
    ch_names: list[str],
    orientation: str,
) -> np.ndarray:
    """Return data as channels by samples."""
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D signal array, got shape {data.shape}")

    orientation = orientation.lower()
    n_samples = timestamps.size
    n_channels = len(ch_names)

    if orientation == "channels_samples":
        standardized = data
    elif orientation == "samples_channels":
        standardized = data.T
    elif orientation == "auto":
        if data.shape == (n_samples, n_channels):
            standardized = data.T
        elif data.shape == (n_channels, n_samples):
            standardized = data
        else:
            raise ValueError(
                "Cannot infer input orientation: "
                f"data shape={data.shape}, n_samples={n_samples}, n_channels={n_channels}"
            )
    else:
        raise ValueError(
            "input_orientation must be 'auto', 'samples_channels', or 'channels_samples'"
        )

    assert standardized.shape == (n_channels, n_samples)
    return np.asarray(standardized, dtype=float)


# loading_data: read the baseline CSV, extract signal columns and convert to volts.
def load_csv_recording(
    input_path: Path,
    config: PreprocessingConfig,
) -> BaselineRecording:
    """Load one CSV baseline recording and convert signal amplitudes to volts."""
    LOGGER.info("Loading %s", input_path)
    df = pd.read_csv(input_path, sep=config.csv_separator)
    df.columns = [str(column).strip() for column in df.columns]

    if config.timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {config.timestamp_column}")

    timestamps = pd.to_numeric(df[config.timestamp_column], errors="coerce").to_numpy(float)
    signal_columns = [
        column
        for column in df.columns
        if column != config.timestamp_column and not is_ignored_column(column, config)
    ]
    if not signal_columns:
        raise ValueError("No EEG/EOG signal columns were found after ignoring metadata columns.")

    signal_df = df[signal_columns].apply(pd.to_numeric, errors="coerce")
    data = signal_df.to_numpy(dtype=float)
    data = standardize_orientation(
        data=data,
        timestamps=timestamps,
        ch_names=signal_columns,
        orientation=config.input_orientation,
    ).copy()
    data *= amplitude_scale_to_volts(config.amplitude_unit)

    duration = float(timestamps[-1] - timestamps[0])
    effective_fs = infer_sampling_frequency(timestamps)
    return BaselineRecording(
        data=data,
        timestamps=timestamps,
        ch_names=signal_columns,
        source_path=input_path,
        original_duration=duration,
        effective_fs=effective_fs,
        metadata={
            "input_shape_after_standardization": list(data.shape),
            "amplitude_unit": config.amplitude_unit,
            "ignored_columns": [
                column for column in df.columns if is_ignored_column(column, config)
            ],
        },
    )


def load_baseline_recording(
    input_path: Path,
    config: PreprocessingConfig,
) -> BaselineRecording:
    """Dispatch to a supported baseline loader; currently only CSV is implemented."""
    validate_config(config)
    suffix = input_path.suffix.lower()
    supported = tuple(ext.lower() for ext in config.input_file_extensions)
    if suffix not in supported or suffix != ".csv":
        supported_text = ", ".join(sorted(set(supported) & {".csv"})) or ".csv"
        raise PreprocessingError(
            f"Unsupported input format {input_path.suffix or '<no suffix>'!r} for "
            f"'{input_path.name}'. Supported input is currently: {supported_text}."
        )
    return load_csv_recording(input_path, config)


# validation: check timestamps, empty/invalid signals and effective sampling frequency.
def validate_recording(
    recording: BaselineRecording,
    config: PreprocessingConfig,
) -> ValidationReport:
    """Validate timestamps and signal shape before preprocessing."""
    data = recording.data
    timestamps = recording.timestamps
    assert data.shape == (len(recording.ch_names), timestamps.size)

    warnings_out: list[str] = []
    if data.shape[1] == 0 or data.shape[0] == 0:
        raise ValueError("Recording has zero channels or zero samples.")

    if np.isnan(data).any():
        warnings_out.append("Signal contains NaN values; affected EEG channels may be marked bad.")
    if np.isinf(data).any():
        warnings_out.append("Signal contains infinite values; affected EEG channels may be marked bad.")

    if np.isnan(timestamps).any() or np.isinf(timestamps).any():
        raise ValueError("Timestamps contain NaN or infinite values.")

    diffs = np.diff(timestamps)
    duplicated = int(np.sum(diffs == 0))
    if duplicated:
        raise ValueError(f"Timestamps contain {duplicated} duplicated adjacent values.")
    if np.any(diffs < 0):
        raise ValueError("Timestamps are non-monotonic.")

    empty_channels = [
        ch_name
        for ch_name, channel in zip(recording.ch_names, data)
        if channel.size == 0 or np.all(pd.isna(channel))
    ]
    if empty_channels:
        warnings_out.append(f"Empty channels detected: {', '.join(empty_channels)}")

    fs_delta = abs(recording.effective_fs - config.nominal_fs)
    if fs_delta > config.fs_warning_hz:
        message = (
            f"Effective sampling frequency {recording.effective_fs:.6f} Hz differs "
            f"from nominal {config.nominal_fs:.6f} Hz by {fs_delta:.6f} Hz."
        )
        if config.sampling_frequency_mode.strip().lower() == "nominal":
            message += (
                " Strong warning: nominal mode will use the nominal rate for processing; "
                "the signal is not resampled."
            )
        warnings_out.append(message)

    for message in warnings_out:
        LOGGER.warning(message)

    return ValidationReport(
        n_channels=data.shape[0],
        n_samples=data.shape[1],
        duration=recording.original_duration,
        effective_fs=recording.effective_fs,
        warnings=warnings_out,
    )


# channel_separation: detect EOG by name and keep it out of EEG-only operations.
def separate_channel_types(
    ch_names: list[str],
    config: PreprocessingConfig,
) -> ChannelGroups:
    """Separate EOG channels from EEG channels by case-insensitive name matching."""
    token = config.eog_name_token.lower()
    eog_indices = [idx for idx, name in enumerate(ch_names) if token in name.lower()]
    eeg_indices = [idx for idx in range(len(ch_names)) if idx not in set(eog_indices)]
    eeg_ch_names = [ch_names[idx] for idx in eeg_indices]
    eog_ch_names = [ch_names[idx] for idx in eog_indices]
    if not eeg_ch_names:
        raise ValueError("No EEG channels remain after EOG separation.")
    return ChannelGroups(
        eeg_indices=eeg_indices,
        eog_indices=eog_indices,
        eeg_ch_names=eeg_ch_names,
        eog_ch_names=eog_ch_names,
    )


# mne_raw_creation: create RawArray, assign channel types and apply the montage.
def create_raw(
    recording: BaselineRecording,
    groups: ChannelGroups,
    config: PreprocessingConfig,
    processing_fs: float,
) -> tuple[Any, list[str], list[str]]:
    """Create an MNE RawArray and apply the configured montage."""
    mne = require_module("mne")
    ch_types = ["eeg"] * len(recording.ch_names)
    for idx in groups.eog_indices:
        ch_types[idx] = "eog"

    info = mne.create_info(
        ch_names=recording.ch_names,
        sfreq=processing_fs,
        ch_types=ch_types,
    )
    raw = mne.io.RawArray(recording.data.copy(), info, verbose=False)

    unmatched_eeg: list[str] = []
    montage_warnings: list[str] = []
    if config.montage:
        montage = mne.channels.make_standard_montage(config.montage)
        montage_lookup = {name.lower() for name in montage.ch_names}
        unmatched_eeg = [
            name for name in groups.eeg_ch_names if name.lower() not in montage_lookup
        ]
        if unmatched_eeg:
            LOGGER.warning(
                "EEG channels not matched by montage %s: %s",
                config.montage,
                ", ".join(unmatched_eeg),
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw.set_montage(
                montage,
                on_missing=config.montage_on_missing,
                match_case=False,
                verbose=False,
            )
        for warning in caught:
            message = f"MNE montage warning: {warning.message}"
            montage_warnings.append(message)
            LOGGER.warning(message)

    assert raw.get_data().shape == recording.data.shape
    return raw, unmatched_eeg, montage_warnings


# detrending: remove constant offsets from continuous EEG and EOG channels.
def detrend_channels(
    data: np.ndarray,
    channel_indices: list[int],
    config: PreprocessingConfig,
) -> np.ndarray:
    """Detrend selected channels along the sample dimension."""
    scipy_signal = require_module("scipy.signal")
    detrend_type = "linear" if config.enable_linear_detrend else config.detrend_type
    if detrend_type != "constant" and not config.enable_linear_detrend:
        raise ValueError("Only constant detrending is enabled unless linear detrending is requested.")

    out = data.copy()
    if channel_indices:
        out[channel_indices] = scipy_signal.detrend(
            out[channel_indices],
            axis=-1,
            type=detrend_type,
        )
    assert out.shape == data.shape
    return out


# flat_segment_check: helper for finding long constant sections in one channel.
def longest_constant_run_seconds(
    channel: np.ndarray,
    fs: float,
    tolerance: float,
) -> float:
    """Return the longest nearly constant segment duration for one channel."""
    finite = np.asarray(channel[np.isfinite(channel)], dtype=float)
    if finite.size < 2:
        return math.inf

    constant_diffs = np.abs(np.diff(finite)) <= tolerance
    longest = 0
    current = 0
    for is_constant in constant_diffs:
        if is_constant:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    if longest == 0:
        return 0.0
    return float((longest + 1) / fs)


# bad_channel_detection: find invalid, zero-variance, low-variance or flat EEG channels.
def detect_bad_channels(
    data: np.ndarray,
    groups: ChannelGroups,
    config: PreprocessingConfig,
    processing_fs: float,
) -> dict[str, list[str]]:
    """Detect invalid, dead, flat, or near-flat EEG channels."""
    bad_reasons: dict[str, list[str]] = {}
    for idx, ch_name in zip(groups.eeg_indices, groups.eeg_ch_names):
        channel = data[idx]
        reasons: list[str] = []
        if np.isnan(channel).any() or np.isinf(channel).any():
            reasons.append("invalid_values")

        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            reasons.append("empty_or_all_invalid")
        else:
            variance = float(np.var(finite))
            if variance == config.zero_variance_threshold:
                reasons.append("zero_variance")
            elif variance <= config.low_variance_threshold:
                reasons.append(f"low_variance={variance:.3e}")

            constant_seconds = longest_constant_run_seconds(
                finite,
                fs=processing_fs,
                tolerance=config.flat_diff_tolerance,
            )
            if constant_seconds >= config.max_constant_segment_seconds:
                reasons.append(f"long_constant_segment={constant_seconds:.3f}s")

        if reasons:
            bad_reasons[ch_name] = reasons

    return bad_reasons


# montage_location_check: interpolation requires valid sensor positions.
def has_valid_eeg_locations(raw: Any, ch_names: list[str]) -> bool:
    """Check whether all requested EEG channels have finite electrode locations."""
    for ch_name in ch_names:
        idx = raw.ch_names.index(ch_name)
        loc = raw.info["chs"][idx]["loc"][:3]
        if not np.all(np.isfinite(loc)) or np.allclose(loc, 0.0):
            return False
    return True


# interpolation: repair bad EEG channels only within the configured 15/20 percent limits.
def interpolate_bad_channels(
    raw: Any,
    bad_reasons: dict[str, list[str]],
    groups: ChannelGroups,
    config: PreprocessingConfig,
) -> tuple[Any, list[str], list[str]]:
    """Interpolate bad EEG channels with MNE spherical interpolation."""
    bad_chs = list(bad_reasons)
    if not bad_chs:
        return raw, [], []

    bad_fraction = len(bad_chs) / len(groups.eeg_ch_names)
    warnings_out: list[str] = []

    if bad_fraction > config.max_interp_fraction_hard:
        raise PreprocessingError(
            f"{len(bad_chs)} bad EEG channels ({bad_fraction:.1%}) exceeds the "
            f"hard interpolation limit of {config.max_interp_fraction_hard:.0%}."
        )
    if bad_fraction > config.max_interp_fraction_soft:
        warning = (
            f"{len(bad_chs)} bad EEG channels ({bad_fraction:.1%}) exceeds the "
            f"soft interpolation limit of {config.max_interp_fraction_soft:.0%}; "
            "interpolating but flagging for strong review."
        )
        warnings_out.append(warning)
        LOGGER.warning(warning)

    if not has_valid_eeg_locations(raw, groups.eeg_ch_names):
        raise PreprocessingError(
            "Cannot interpolate bad EEG channels because the EEG montage does not "
            "provide valid locations for every EEG channel."
        )

    raw_interp = raw.copy()
    data = raw_interp.get_data()
    for ch_name in bad_chs:
        idx = raw_interp.ch_names.index(ch_name)
        data[idx] = 0.0
    raw_interp._data = data
    raw_interp.info["bads"] = bad_chs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_interp.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    for warning in caught:
        message = f"MNE interpolation warning: {warning.message}"
        warnings_out.append(message)
        LOGGER.warning(message)
    assert raw_interp.get_data().shape == raw.get_data().shape
    return raw_interp, bad_chs, warnings_out


# notch_filter: fallback line-noise cleanup used only when ZapLine is disabled or fails.
def notch_filter_array(
    data: np.ndarray,
    fs: float,
    fline: float,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply a zero-phase fallback notch filter to channel-by-sample data."""
    scipy_signal = require_module("scipy.signal")
    b, a = scipy_signal.iirnotch(
        w0=fline,
        Q=config.notch_quality_factor,
        fs=fs,
    )
    padlen = config.filter_padlen
    if padlen is not None and data.shape[-1] <= padlen:
        raise PreprocessingError(
            f"Cannot apply notch filter at {fline} Hz: {data.shape[-1]} samples "
            f"is not greater than padlen={padlen}."
        )
    return scipy_signal.filtfilt(b, a, data, axis=-1, padlen=padlen)


# zapline: remove 50 Hz line noise using meegkit.dss.dss_line.
def apply_zapline_to_indices(
    data: np.ndarray,
    channel_indices: list[int],
    fs: float,
    nremove: int,
    config: PreprocessingConfig,
    label: str,
    enabled: bool | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply ZapLine to selected channels, with optional notch fallback."""
    out = data.copy()
    zapline_enabled = config.zapline_enabled if enabled is None else enabled
    report: dict[str, Any] = {
        "label": label,
        "zapline_enabled": zapline_enabled,
        "processing_skipped": False,
        "skip_reason": None,
        "processed_harmonics": [],
        "skipped_harmonics": [],
        "fallback_notch_harmonics": [],
    }
    if not channel_indices:
        report["processing_skipped"] = True
        report["skip_reason"] = "no_channels"
        return out, report

    harmonics = [
        config.line_noise_hz * harmonic
        for harmonic in config.zap_harmonics
        if config.line_noise_hz * harmonic <= config.eeg_high_cut_hz
    ]
    report["skipped_harmonics"] = [
        config.line_noise_hz * harmonic
        for harmonic in config.zap_harmonics
        if config.line_noise_hz * harmonic > config.eeg_high_cut_hz
    ]

    if not harmonics:
        return out, report

    if zapline_enabled:
        try:
            from meegkit.dss import dss_line

            segment = out[channel_indices].T[:, :, np.newaxis]
            for fline in harmonics:
                segment, _ = dss_line(segment, fline=fline, sfreq=fs, nremove=nremove)
                report["processed_harmonics"].append(fline)
            out[channel_indices] = np.squeeze(segment, axis=2).T
            assert out.shape == data.shape
            return out, report
        except Exception as exc:  # noqa: BLE001 - fallback needs the original failure.
            report["zapline_error"] = str(exc)
            LOGGER.warning("ZapLine failed for %s: %s", label, exc)
            if not config.fallback_notch_enabled:
                raise PreprocessingError(f"ZapLine failed for {label} and notch fallback is disabled.") from exc

    if config.fallback_notch_enabled:
        for fline in harmonics:
            out[channel_indices] = notch_filter_array(
                out[channel_indices],
                fs=fs,
                fline=fline,
                config=config,
            )
            report["fallback_notch_harmonics"].append(fline)
    assert out.shape == data.shape
    return out, report


# power_line_noise_removal: process EEG and EOG separately with different nremove defaults.
def apply_zapline(
    data: np.ndarray,
    groups: ChannelGroups,
    config: PreprocessingConfig,
    processing_fs: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Process EEG line noise and optionally process multi-channel EOG line noise."""
    out, eeg_report = apply_zapline_to_indices(
        data=data,
        channel_indices=groups.eeg_indices,
        fs=processing_fs,
        nremove=config.zap_nremove_eeg,
        config=config,
        label="EEG",
    )

    eog_report: dict[str, Any] = {
        "label": "EOG",
        "zapline_enabled": config.zapline_eog_enabled,
        "processing_skipped": True,
        "skip_reason": None,
        "processed_harmonics": [],
        "skipped_harmonics": [],
        "fallback_notch_harmonics": [],
    }
    if not groups.eog_indices:
        eog_report["skip_reason"] = "no_eog_channels"
    elif not config.zapline_eog_enabled:
        eog_report["skip_reason"] = "disabled_by_configuration"
    elif len(groups.eog_indices) == 1:
        warning = (
            "EOG ZapLine was requested but skipped because DSS/ZapLine requires "
            "more than one EOG channel; the single EOG channel remains unchanged "
            "until the 10 Hz low-pass filter."
        )
        LOGGER.warning(warning)
        eog_report["skip_reason"] = "single_eog_channel"
        eog_report["warning"] = warning
    else:
        out, eog_report = apply_zapline_to_indices(
            data=out,
            channel_indices=groups.eog_indices,
            fs=processing_fs,
            nremove=config.zap_nremove_eog,
            config=config,
            label="EOG",
            enabled=config.zapline_eog_enabled,
        )

    assert out.shape == data.shape
    return out, {"eeg": eeg_report, "eog": eog_report}


# filter_safety: sosfiltfilt wrapper with explicit short-recording errors.
def sosfiltfilt_checked(
    data: np.ndarray,
    sos: np.ndarray,
    config: PreprocessingConfig,
    stage: str,
) -> np.ndarray:
    """Apply zero-phase SOS filtering with an informative short-recording error."""
    scipy_signal = require_module("scipy.signal")
    padlen = config.filter_padlen
    if padlen is not None and data.shape[-1] <= padlen:
        raise PreprocessingError(
            f"{stage} cannot be applied: {data.shape[-1]} samples is not greater "
            f"than padlen={padlen}."
        )
    try:
        filtered = scipy_signal.sosfiltfilt(sos, data, axis=-1, padlen=padlen)
    except ValueError as exc:
        raise PreprocessingError(f"{stage} failed, likely because the recording is too short: {exc}") from exc
    assert filtered.shape == data.shape, (
        f"{stage} changed the signal shape from {data.shape} to {filtered.shape}."
    )
    return filtered


# eeg_bandpass_filter: fourth-order 0.5-80 Hz zero-phase Butterworth filtering.
def filter_eeg(
    data: np.ndarray,
    groups: ChannelGroups,
    config: PreprocessingConfig,
    processing_fs: float,
) -> np.ndarray:
    """Apply fourth-order zero-phase Butterworth band-pass filtering to EEG."""
    if not groups.eeg_indices:
        return data
    scipy_signal = require_module("scipy.signal")
    sos = scipy_signal.butter(
        config.butterworth_order,
        [config.eeg_low_cut_hz, config.eeg_high_cut_hz],
        btype="bandpass",
        fs=processing_fs,
        output="sos",
    )
    out = data.copy()
    filtered_eeg = sosfiltfilt_checked(
        out[groups.eeg_indices],
        sos=sos,
        config=config,
        stage="EEG band-pass filter",
    )
    assert filtered_eeg.shape == out[groups.eeg_indices].shape
    out[groups.eeg_indices] = filtered_eeg
    assert out.shape == data.shape
    return out


# eog_filter: fourth-order 10 Hz zero-phase low-pass filtering for EOG.
def filter_eog(
    data: np.ndarray,
    groups: ChannelGroups,
    config: PreprocessingConfig,
    processing_fs: float,
) -> np.ndarray:
    """Apply fourth-order zero-phase low-pass filtering to EOG channels."""
    if not groups.eog_indices:
        return data
    scipy_signal = require_module("scipy.signal")
    sos = scipy_signal.butter(
        config.butterworth_order,
        config.eog_lowpass_hz,
        btype="lowpass",
        fs=processing_fs,
        output="sos",
    )
    out = data.copy()
    filtered_eog = sosfiltfilt_checked(
        out[groups.eog_indices],
        sos=sos,
        config=config,
        stage="EOG low-pass filter",
    )
    assert filtered_eog.shape == out[groups.eog_indices].shape
    out[groups.eog_indices] = filtered_eog
    assert out.shape == data.shape
    return out


# crop_window: compute the centered or user-configured 60-second interval.
def compute_crop_window(
    timestamps: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[float, float]:
    """Compute the requested crop interval in timestamp units."""
    first = float(timestamps[0])
    last = float(timestamps[-1])
    duration = last - first
    if duration < config.crop_duration_seconds:
        raise InsufficientDurationError(
            f"Recording duration {duration:.6f}s is shorter than "
            f"{config.crop_duration_seconds:.6f}s."
        )

    if config.crop_start_time is not None and config.crop_start_offset_seconds is not None:
        raise ValueError("Use only one of crop_start_time or crop_start_offset_seconds.")
    if config.crop_start_time is not None:
        crop_start = float(config.crop_start_time)
    elif config.crop_start_offset_seconds is not None:
        crop_start = first + float(config.crop_start_offset_seconds)
    else:
        crop_start = first + (duration - config.crop_duration_seconds) / 2.0

    crop_end = crop_start + config.crop_duration_seconds
    if crop_start < first or crop_end > last:
        raise InsufficientDurationError(
            f"Requested crop [{crop_start:.6f}, {crop_end:.6f}] is outside "
            f"recording bounds [{first:.6f}, {last:.6f}]."
        )
    return crop_start, crop_end


# sixty_second_crop: crop after filtering and validate duration from sample coverage.
def crop_recording(
    data: np.ndarray,
    timestamps: np.ndarray,
    config: PreprocessingConfig,
    processing_fs: float,
) -> CropResult:
    """Crop the configured 60-second interval after filtering."""
    crop_start, crop_end = compute_crop_window(timestamps, config)
    expected_samples = int(round(config.crop_duration_seconds * processing_fs))
    if expected_samples <= 0:
        raise ValueError("Target crop sample count must be positive.")

    if config.crop_enforce_nominal_sample_count:
        if timestamps.size < expected_samples:
            raise InsufficientDurationError(
                f"Recording has {timestamps.size} samples but {expected_samples} are needed "
                "for the requested processing-duration crop."
            )
        start_idx = int(np.searchsorted(timestamps, crop_start, side="left"))
        if start_idx + expected_samples > timestamps.size:
            start_idx = timestamps.size - expected_samples
        end_idx = start_idx + expected_samples
        indices = np.arange(start_idx, end_idx)
    else:
        indices = np.flatnonzero((timestamps >= crop_start) & (timestamps < crop_end))
        if indices.size == 0:
            raise InsufficientDurationError("Timestamp crop produced no samples.")

    cropped_data = data[:, indices].copy()
    cropped_timestamps = timestamps[indices].copy()
    retained_samples = int(cropped_data.shape[1])
    sample_count_error = abs(retained_samples - expected_samples)
    if sample_count_error > config.crop_sample_count_tolerance:
        raise PreprocessingError(
            f"Crop retained {retained_samples} samples; expected {expected_samples} "
            f"at {processing_fs:.6f} Hz (tolerance "
            f"{config.crop_sample_count_tolerance} sample(s))."
        )

    first_to_last_span = float(cropped_timestamps[-1] - cropped_timestamps[0])
    sample_coverage_duration = retained_samples / processing_fs
    expected_first_to_last_span = max(0.0, (retained_samples - 1) / processing_fs)
    if (
        abs(first_to_last_span - expected_first_to_last_span)
        > config.crop_duration_tolerance_seconds
    ):
        raise PreprocessingError(
            f"Crop timestamp spacing is inconsistent with {processing_fs:.6f} Hz: "
            f"first-to-last span is {first_to_last_span:.6f}s, expected approximately "
            f"{expected_first_to_last_span:.6f}s within "
            f"{config.crop_duration_tolerance_seconds:.6f}s."
        )

    assert cropped_data.shape[1] == cropped_timestamps.size
    return CropResult(
        data=cropped_data,
        timestamps=cropped_timestamps,
        crop_start=crop_start,
        crop_end=crop_end,
        requested_duration=config.crop_duration_seconds,
        actual_timestamp_span=first_to_last_span,
        n_samples=retained_samples,
        expected_samples=expected_samples,
        sample_coverage_duration=sample_coverage_duration,
        first_to_last_timestamp_span=first_to_last_span,
    )


# average_reference: apply common-average reference to EEG channels only.
def apply_average_reference(
    data: np.ndarray,
    groups: ChannelGroups,
) -> np.ndarray:
    """Apply common-average reference to EEG channels only."""
    out = data.copy()
    eeg_mean = np.mean(out[groups.eeg_indices], axis=0, keepdims=True)
    out[groups.eeg_indices] = out[groups.eeg_indices] - eeg_mean
    assert out.shape == data.shape
    return out


# fixed_length_epoching: split the cleaned 60-second recording into 4-second epochs.
def create_fixed_epochs(
    data: np.ndarray,
    timestamps: np.ndarray,
    groups: ChannelGroups,
    config: PreprocessingConfig,
    processing_fs: float,
) -> EpochResult:
    """Create fixed-length non-overlapping EEG and EOG epochs."""
    samples_per_epoch = int(round(config.epoch_duration_seconds * processing_fs))
    overlap_samples = int(round(config.epoch_overlap_seconds * processing_fs))
    if samples_per_epoch <= 0:
        raise ValueError("Epoch duration yields zero samples.")
    if overlap_samples < 0 or overlap_samples >= samples_per_epoch:
        raise ValueError("Epoch overlap must be >= 0 and shorter than the epoch duration.")

    step = samples_per_epoch - overlap_samples
    starts = list(range(0, data.shape[1] - samples_per_epoch + 1, step))
    if not starts:
        raise PreprocessingError(
            f"No complete {config.epoch_duration_seconds:.3f}s epochs can be created "
            f"from {data.shape[1]} samples."
        )

    eeg_epochs = np.stack(
        [data[groups.eeg_indices, start : start + samples_per_epoch] for start in starts],
        axis=0,
    )
    if groups.eog_indices:
        eog_epochs = np.stack(
            [data[groups.eog_indices, start : start + samples_per_epoch] for start in starts],
            axis=0,
        )
    else:
        eog_epochs = None

    epoch_start_times = np.asarray([timestamps[start] for start in starts], dtype=float)
    epoch_end_times = epoch_start_times + config.epoch_duration_seconds

    assert eeg_epochs.shape == (len(starts), len(groups.eeg_indices), samples_per_epoch)
    if eog_epochs is not None:
        assert eog_epochs.shape == (len(starts), len(groups.eog_indices), samples_per_epoch)
    return EpochResult(
        eeg_epochs=eeg_epochs,
        eog_epochs=eog_epochs,
        epoch_start_times=epoch_start_times,
        epoch_end_times=epoch_end_times,
        samples_per_epoch=samples_per_epoch,
    )


# epoch_rejection_threshold: support either absolute count or fraction of bad channels.
def resolve_max_bad_channels(max_bad_channels: int | float, n_channels: int) -> int:
    """Resolve an absolute or fractional bad-channel epoch threshold."""
    if isinstance(max_bad_channels, float) and 0 < max_bad_channels < 1:
        return max(1, int(math.ceil(max_bad_channels * n_channels)))
    return int(max_bad_channels)


# bad_epoch_rejection: reject epochs with invalid data, extreme amplitude or too many bad channels.
def reject_bad_epochs(
    eeg_epochs: np.ndarray,
    ch_names: list[str],
    config: PreprocessingConfig,
) -> RejectionResult:
    """Reject epochs according to documented hybrid quality criteria."""
    max_bad = resolve_max_bad_channels(config.max_bad_channels, len(ch_names))
    retained: list[int] = []
    rejected: list[int] = []
    reasons_by_epoch: dict[int, list[str]] = {}
    quality_details: dict[int, dict[str, Any]] = {}

    for epoch_idx, epoch in enumerate(eeg_epochs):
        invalid_mask = ~np.isfinite(epoch)
        invalid_channels = np.flatnonzero(np.any(invalid_mask, axis=1))

        finite_epoch = np.nan_to_num(epoch, nan=0.0, posinf=0.0, neginf=0.0)
        ptp = np.ptp(finite_epoch, axis=1)
        max_abs = np.max(np.abs(finite_epoch), axis=1)
        variance = np.var(finite_epoch, axis=1)

        extreme_channels = np.flatnonzero(
            ptp > config.epoch_extreme_peak_to_peak_threshold_volts
        )
        high_ptp_channels = np.flatnonzero(
            ptp > config.epoch_peak_to_peak_threshold_volts
        )
        excursion_channels = np.flatnonzero(
            max_abs > config.epoch_large_excursion_threshold_volts
        )
        flat_channels = np.flatnonzero(
            (variance <= config.epoch_flat_variance_threshold)
            | (ptp <= config.epoch_flat_peak_to_peak_threshold_volts)
        )

        category_indices = {
            "invalid_values": invalid_channels,
            "high_peak_to_peak": high_ptp_channels,
            "extreme_peak_to_peak": extreme_channels,
            "large_absolute_excursion": excursion_channels,
            "flat_signal": flat_channels,
        }
        contaminated_indices = sorted(
            {
                int(channel_idx)
                for indices in category_indices.values()
                for channel_idx in indices
            }
        )
        contaminated_count = len(contaminated_indices)
        limit_exceeded = contaminated_count > max_bad
        reject = bool(invalid_channels.size or extreme_channels.size or limit_exceeded)

        details = {
            category: [ch_names[int(idx)] for idx in indices]
            for category, indices in category_indices.items()
        }
        details.update(
            {
                "contaminated_channels": [ch_names[idx] for idx in contaminated_indices],
                "contaminated_channel_count": contaminated_count,
                "max_bad_channels": max_bad,
                "contaminated_channel_limit_exceeded": limit_exceeded,
                "rejected": reject,
            }
        )
        quality_details[epoch_idx] = details

        if reject:
            rejected.append(epoch_idx)
            reasons = [
                f"{category}: {', '.join(names)}"
                for category, names in (
                    (category, details[category]) for category in category_indices
                )
                if names
            ]
            reasons.append(f"contaminated_channel_count: {contaminated_count}")
            if limit_exceeded:
                reasons.append(
                    f"too_many_contaminated_channels: {contaminated_count} > {max_bad}"
                )
            else:
                reasons.append(
                    "contaminated_channel_limit_exceeded: false "
                    f"({contaminated_count} <= {max_bad})"
                )
            reasons_by_epoch[epoch_idx] = reasons
        else:
            retained.append(epoch_idx)

    return RejectionResult(
        retained_indices=retained,
        rejected_indices=rejected,
        rejection_reasons=reasons_by_epoch,
        epoch_quality_details=quality_details,
    )


# filter_metadata: collect processing parameters for NPZ and JSON provenance.
def make_filter_parameters(
    config: PreprocessingConfig,
    zapline_report: dict[str, Any],
    processing_fs: float,
) -> dict[str, Any]:
    """Collect filter settings for NPZ and JSON output."""
    return {
        "detrend_type": "linear" if config.enable_linear_detrend else config.detrend_type,
        "zapline": zapline_report,
        "butterworth_order": config.butterworth_order,
        "eeg_bandpass_hz": [config.eeg_low_cut_hz, config.eeg_high_cut_hz],
        "eog_lowpass_hz": config.eog_lowpass_hz,
        "filter_padlen": config.filter_padlen,
        "fs_used_for_filtering": processing_fs,
    }


# serialization_helpers: convert nested objects into JSON-safe report fields.
def json_dumps_compact(value: Any) -> str:
    """Serialize nested metadata for storage in NPZ string fields."""
    return json.dumps(to_jsonable(value), sort_keys=True)


def to_jsonable(value: Any) -> Any:
    """Convert numpy, pathlib, and dataclass values into JSON-safe objects."""
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


# save_output: write the final KPLS-ready EEG epochs and aligned metadata.
def save_processed_recording(
    output_dir: Path,
    input_path: Path,
    epochs: EpochResult,
    rejection: RejectionResult,
    groups: ChannelGroups,
    crop: CropResult,
    original_duration: float,
    interpolated_channels: list[str],
    filter_parameters: dict[str, Any],
    quality_status: str,
    config: PreprocessingConfig,
    processing_fs: float,
) -> Path:
    """Save compressed NPZ output for one processed recording."""
    output_dir.mkdir(parents=True, exist_ok=True)
    retained = rejection.retained_indices
    eeg_retained = epochs.eeg_epochs[retained]
    has_eog = epochs.eog_epochs is not None
    eog_retained = (
        epochs.eog_epochs[retained]
        if has_eog
        else np.empty((0, 0, 0), dtype=float)
    )
    output_path = output_dir / f"{input_path.stem}_preprocessed.npz"

    payload = {
        "data": np.asarray(eeg_retained, dtype=float),
        "eog_data": np.asarray(eog_retained, dtype=float),
        "has_eog": np.asarray(has_eog, dtype=bool),
        "fs": np.asarray(processing_fs, dtype=float),
        "ch_names": np.asarray(groups.eeg_ch_names, dtype=str),
        "eog_ch_names": np.asarray(groups.eog_ch_names, dtype=str),
        "epoch_start_times": np.asarray(epochs.epoch_start_times[retained], dtype=float),
        "epoch_end_times": np.asarray(epochs.epoch_end_times[retained], dtype=float),
        "interpolated_channels": np.asarray(interpolated_channels, dtype=str),
        "rejected_epoch_indices": np.asarray(rejection.rejected_indices, dtype=int),
        "rejection_reasons": np.asarray(json_dumps_compact(rejection.rejection_reasons)),
        "filter_parameters": np.asarray(json_dumps_compact(filter_parameters)),
        "crop_start": np.asarray(crop.crop_start, dtype=float),
        "crop_end": np.asarray(crop.crop_end, dtype=float),
        "expected_crop_samples": np.asarray(crop.expected_samples, dtype=int),
        "retained_crop_samples": np.asarray(crop.n_samples, dtype=int),
        "sample_coverage_duration": np.asarray(crop.sample_coverage_duration, dtype=float),
        "first_to_last_timestamp_span": np.asarray(
            crop.first_to_last_timestamp_span,
            dtype=float,
        ),
        "original_duration": np.asarray(original_duration, dtype=float),
        "processed_duration": np.asarray(crop.requested_duration, dtype=float),
        "quality_status": np.asarray(quality_status),
    }
    object_fields = [name for name, value in payload.items() if value.dtype == object]
    if object_fields:
        raise PreprocessingError(
            "Refusing to save object arrays in NPZ: " + ", ".join(object_fields)
        )
    np.savez_compressed(output_path, **payload)
    return output_path


# qc_report: write a per-recording JSON summary of validation and quality decisions.
def generate_qc_report(
    output_dir: Path,
    input_path: Path,
    validation: ValidationReport,
    groups: ChannelGroups,
    crop: CropResult | None,
    interpolated_channels: list[str],
    bad_channel_reasons: dict[str, list[str]],
    rejection: RejectionResult | None,
    output_path: Path | None,
    quality_status: str,
    config: PreprocessingConfig,
    processing_fs: float,
    warning_messages: list[str] | None = None,
    unmatched_montage_channels: list[str] | None = None,
    zapline_report: dict[str, Any] | None = None,
    plot_paths: list[Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save compact QC metadata without raw, filtered, cropped, or epoched arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{input_path.stem}_qc.json"
    crop_metadata = None
    if crop is not None:
        crop_metadata = {
            "crop_start": crop.crop_start,
            "crop_end": crop.crop_end,
            "requested_duration": crop.requested_duration,
            "expected_samples": crop.expected_samples,
            "n_samples": crop.n_samples,
            "sample_coverage_duration": crop.sample_coverage_duration,
            "first_to_last_timestamp_span": crop.first_to_last_timestamp_span,
            # Kept as a compatibility label; it contains metadata, never signal data.
            "actual_timestamp_span": crop.actual_timestamp_span,
        }

    total_epochs = (
        len(rejection.retained_indices) + len(rejection.rejected_indices)
        if rejection
        else 0
    )
    report = {
        "input_path": str(input_path),
        "quality_status": quality_status,
        "output_path": str(output_path) if output_path else None,
        "output_paths": {
            "npz": str(output_path) if output_path else None,
            "qc_json": str(report_path),
            "plots": [str(path) for path in (plot_paths or [])],
        },
        "sampling_frequency": {
            "nominal_hz": config.nominal_fs,
            "effective_hz": validation.effective_fs,
            "processing_hz": processing_fs,
            "mode": config.sampling_frequency_mode,
        },
        "original_duration": validation.duration,
        "validation": validation,
        "n_eeg_channels": len(groups.eeg_ch_names),
        "n_eog_channels": len(groups.eog_ch_names),
        "eeg_ch_names": groups.eeg_ch_names,
        "eog_ch_names": groups.eog_ch_names,
        "bad_channel_reasons": bad_channel_reasons,
        "interpolated_channels": interpolated_channels,
        "unmatched_montage_channels": unmatched_montage_channels or [],
        "crop": crop_metadata,
        "zapline": zapline_report or {},
        "eog_line_noise_processing": (zapline_report or {}).get("eog", {}),
        "total_generated_epochs": total_epochs,
        "rejected_epoch_indices": rejection.rejected_indices if rejection else [],
        "rejection_reasons": rejection.rejection_reasons if rejection else {},
        "epoch_quality_details": rejection.epoch_quality_details if rejection else {},
        "retained_epoch_count": len(rejection.retained_indices) if rejection else 0,
        "warning_messages": warning_messages or [],
        "config": config,
        "extra": extra or {},
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(report), handle, indent=2, sort_keys=True)
    return report_path


# qc_plots: optional non-blocking diagnostic figures for manual inspection.
def generate_qc_plots(
    output_dir: Path,
    input_path: Path,
    raw_before: np.ndarray,
    processed_crop: np.ndarray,
    groups: ChannelGroups,
    config: PreprocessingConfig,
    processing_fs: float,
    all_eeg_epochs: np.ndarray,
    eeg_ch_names: list[str],
    rejected_epoch_indices: list[int],
    rejection_reasons: dict[int, list[str]],
    bad_channel_reasons: dict[str, list[str]],
) -> tuple[list[Path], list[str]]:
    """Generate non-blocking trace, PSD, flag, and rejected-epoch plots."""
    if not config.generate_plots:
        return [], []

    try:
        matplotlib = require_module("matplotlib")
        matplotlib.use("Agg")
        pyplot = require_module("matplotlib.pyplot")
        scipy_signal = require_module("scipy.signal")
    except PreprocessingError as exc:
        message = f"QC plots were requested but unavailable: {exc}"
        LOGGER.warning(message)
        return [], [message]

    saved: list[Path] = []
    plot_warnings: list[str] = []
    plot_dir = output_dir / "plots"
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - plots must not stop batch processing.
        message = f"Could not create QC plot directory '{plot_dir}': {exc}"
        LOGGER.warning(message)
        return [], [message]

    def record_plot_failure(label: str, exc: Exception) -> None:
        message = f"QC plot '{label}' failed: {exc}"
        plot_warnings.append(message)
        LOGGER.warning(message)

    eeg_indices = groups.eeg_indices[: config.plot_channel_count]
    labels = groups.eeg_ch_names[: len(eeg_indices)]

    fig = None
    try:
        time_before = np.arange(raw_before.shape[1]) / processing_fs
        time_after = np.arange(processed_crop.shape[1]) / processing_fs
        fig, axes = pyplot.subplots(2, 1, figsize=(12, 8), sharex=False)
        for local_idx, channel_idx in enumerate(eeg_indices):
            axes[0].plot(
                time_before,
                raw_before[channel_idx] * 1e6,
                linewidth=0.6,
                label=labels[local_idx],
            )
            axes[1].plot(
                time_after,
                processed_crop[channel_idx] * 1e6,
                linewidth=0.6,
                label=labels[local_idx],
            )
        axes[0].set_title("Raw EEG before preprocessing")
        axes[1].set_title("Processed cropped EEG")
        axes[0].set_ylabel("uV")
        axes[1].set_ylabel("uV")
        axes[1].set_xlabel("Seconds")
        axes[0].legend(loc="upper right", fontsize="small")
        axes[1].legend(loc="upper right", fontsize="small")
        fig.tight_layout()
        traces_path = plot_dir / f"{input_path.stem}_traces.png"
        fig.savefig(traces_path, dpi=150)
        saved.append(traces_path)
    except Exception as exc:  # noqa: BLE001 - plots must not stop preprocessing.
        record_plot_failure("traces", exc)
    finally:
        if fig is not None:
            pyplot.close(fig)

    fig = None
    try:
        fig, axes = pyplot.subplots(1, 2, figsize=(12, 4))
        raw_eeg = raw_before[groups.eeg_indices]
        proc_eeg = processed_crop[groups.eeg_indices]
        raw_nperseg = min(4096, raw_eeg.shape[-1])
        proc_nperseg = min(4096, proc_eeg.shape[-1])
        f_raw, p_raw = scipy_signal.welch(
            raw_eeg,
            fs=processing_fs,
            axis=-1,
            nperseg=raw_nperseg,
        )
        f_proc, p_proc = scipy_signal.welch(
            proc_eeg,
            fs=processing_fs,
            axis=-1,
            nperseg=proc_nperseg,
        )
        axes[0].semilogy(f_raw, np.mean(p_raw, axis=0), label="raw")
        axes[0].semilogy(f_proc, np.mean(p_proc, axis=0), label="processed")
        axes[0].set_xlim(0, config.eeg_high_cut_hz + 10)
        axes[0].set_title("PSD before/after")
        axes[0].legend()
        axes[1].semilogy(f_raw, np.mean(p_raw, axis=0), label="raw")
        axes[1].semilogy(f_proc, np.mean(p_proc, axis=0), label="processed")
        axes[1].set_xlim(config.line_noise_hz - 5, config.line_noise_hz + 5)
        axes[1].set_title("PSD around 50 Hz")
        axes[1].legend()
        fig.tight_layout()
        psd_path = plot_dir / f"{input_path.stem}_psd.png"
        fig.savefig(psd_path, dpi=150)
        saved.append(psd_path)
    except Exception as exc:  # noqa: BLE001 - plots must not stop preprocessing.
        record_plot_failure("PSD", exc)
    finally:
        if fig is not None:
            pyplot.close(fig)

    fig = None
    try:
        fig, ax = pyplot.subplots(figsize=(10, 4))
        text = [
            "Bad channels:",
            *(f"{ch}: {', '.join(reasons)}" for ch, reasons in bad_channel_reasons.items()),
            "",
            "Rejected epochs:",
            ", ".join(map(str, rejected_epoch_indices)) if rejected_epoch_indices else "None",
        ]
        ax.axis("off")
        ax.text(0.01, 0.98, "\n".join(text), va="top", family="monospace")
        summary_path = plot_dir / f"{input_path.stem}_quality_flags.png"
        fig.savefig(summary_path, dpi=150)
        saved.append(summary_path)
    except Exception as exc:  # noqa: BLE001 - plots must not stop preprocessing.
        record_plot_failure("quality flags", exc)
    finally:
        if fig is not None:
            pyplot.close(fig)

    rejected_to_plot = rejected_epoch_indices[: config.max_rejected_epoch_plots]
    for epoch_idx in rejected_to_plot:
        if epoch_idx < 0 or epoch_idx >= all_eeg_epochs.shape[0]:
            record_plot_failure(
                f"rejected epoch {epoch_idx}",
                IndexError(f"epoch index is outside 0..{all_eeg_epochs.shape[0] - 1}"),
            )
            continue

        fig = None
        try:
            epoch = all_eeg_epochs[epoch_idx]
            channel_count = min(
                config.rejected_epoch_plot_channel_count,
                epoch.shape[0],
                len(eeg_ch_names),
            )
            time_axis = np.arange(epoch.shape[-1]) / processing_fs
            fig, axes = pyplot.subplots(
                channel_count,
                1,
                figsize=(12, max(4, 1.6 * channel_count)),
                sharex=True,
                squeeze=False,
            )
            for channel_idx in range(channel_count):
                ax = axes[channel_idx, 0]
                ax.plot(time_axis, epoch[channel_idx] * 1e6, linewidth=0.7)
                ax.set_ylabel(f"{eeg_ch_names[channel_idx]}\n(uV)")
            axes[-1, 0].set_xlabel("Seconds")
            concise_reasons = "; ".join(rejection_reasons.get(epoch_idx, []))
            if len(concise_reasons) > 220:
                concise_reasons = concise_reasons[:217] + "..."
            fig.suptitle(
                f"Rejected EEG epoch {epoch_idx:03d} (original index)\n{concise_reasons}",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            rejected_path = plot_dir / (
                f"{input_path.stem}_rejected_epoch_{epoch_idx:03d}.png"
            )
            fig.savefig(rejected_path, dpi=150)
            saved.append(rejected_path)
        except Exception as exc:  # noqa: BLE001 - plots must not stop preprocessing.
            record_plot_failure(f"rejected epoch {epoch_idx}", exc)
        finally:
            if fig is not None:
                pyplot.close(fig)

    return saved, plot_warnings


# epoch_alignment: keep EOG epochs synchronized with retained EEG epoch indices.
def apply_retained_epoch_indices(
    epochs: EpochResult,
    retained_indices: list[int],
) -> EpochResult:
    """Return an EpochResult containing only retained EEG/EOG epochs."""
    return EpochResult(
        eeg_epochs=epochs.eeg_epochs[retained_indices],
        eog_epochs=epochs.eog_epochs[retained_indices] if epochs.eog_epochs is not None else None,
        epoch_start_times=epochs.epoch_start_times[retained_indices],
        epoch_end_times=epochs.epoch_end_times[retained_indices],
        samples_per_epoch=epochs.samples_per_epoch,
    )


# dry_run: validate the file and show planned shapes/operations without saving.
def dry_run_recording(
    input_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
) -> ProcessingSummary:
    """Load and validate a recording, then report proposed operations without saving."""
    recording = load_baseline_recording(input_path, config)
    validation = validate_recording(recording, config)
    processing_fs = determine_processing_fs(recording, config)
    groups = separate_channel_types(recording.ch_names, config)
    if config.eog_required and not groups.eog_indices:
        raise PreprocessingError(
            "No EOG channel was found while eog_required=True; recording requires manual review."
        )
    crop_start, crop_end = compute_crop_window(recording.timestamps, config)

    if config.crop_enforce_nominal_sample_count:
        proposed_crop_samples = int(round(config.crop_duration_seconds * processing_fs))
    else:
        proposed_crop_samples = int(np.sum((recording.timestamps >= crop_start) & (recording.timestamps < crop_end)))
    samples_per_epoch = int(round(config.epoch_duration_seconds * processing_fs))
    overlap_samples = int(round(config.epoch_overlap_seconds * processing_fs))
    step = samples_per_epoch - overlap_samples
    if proposed_crop_samples < samples_per_epoch:
        proposed_epochs = 0
    else:
        proposed_epochs = 1 + (proposed_crop_samples - samples_per_epoch) // step

    LOGGER.info("Dry run only; no files will be saved.")
    LOGGER.info(
        "Sampling-frequency mode=%s; processing_fs=%.6f Hz (nominal=%.6f, effective=%.6f); no resampling.",
        config.sampling_frequency_mode,
        processing_fs,
        config.nominal_fs,
        recording.effective_fs,
    )
    LOGGER.info("Would detrend EEG and EOG with type=%s.", config.detrend_type)
    LOGGER.info("Would detect/interpolate bad EEG channels with %.0f%%/%.0f%% guards.",
                config.max_interp_fraction_soft * 100,
                config.max_interp_fraction_hard * 100)
    LOGGER.info("Would run EEG ZapLine at harmonics <= %.1f Hz, then filter EEG %.1f-%.1f Hz and EOG low-pass %.1f Hz.",
                config.eeg_high_cut_hz,
                config.eeg_low_cut_hz,
                config.eeg_high_cut_hz,
                config.eog_lowpass_hz)
    if not config.zapline_eog_enabled:
        LOGGER.info("Would skip EOG ZapLine and EOG notch fallback by configuration.")
    elif len(groups.eog_indices) == 1:
        LOGGER.warning("Would skip EOG ZapLine because only one EOG channel is present.")
    else:
        LOGGER.info("Would run EOG ZapLine on %d EOG channels.", len(groups.eog_indices))
    LOGGER.info("Would crop [%0.6f, %0.6f] and retain about %d samples.",
                crop_start,
                crop_end,
                proposed_crop_samples)
    LOGGER.info("Would create %d epochs of %d samples each.", proposed_epochs, samples_per_epoch)

    return ProcessingSummary(
        input_path=input_path,
        output_path=None,
        qc_report_path=None,
        original_duration=validation.duration,
        cropped_duration=config.crop_duration_seconds,
        n_eeg_channels=len(groups.eeg_ch_names),
        n_eog_channels=len(groups.eog_ch_names),
        interpolated_channels=[],
        total_epochs=proposed_epochs,
        rejected_epochs=0,
        retained_epochs=proposed_epochs,
        quality_status="DRY_RUN",
    )


# pipeline_orchestration: run all preprocessing stages for one recording.
def process_recording(
    input_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    dry_run: bool = False,
) -> ProcessingSummary:
    """Run the complete preprocessing pipeline for one baseline recording."""
    if dry_run:
        return dry_run_recording(input_path, output_dir, config)

    # loading_data: CSV -> channel-by-sample volts array plus stream_time.
    recording = load_baseline_recording(input_path, config)

    # validation: basic timestamp, shape, NaN/inf and effective-fs checks.
    validation = validate_recording(recording, config)
    processing_fs = determine_processing_fs(recording, config)
    LOGGER.info(
        "Using %.6f Hz for processing (mode=%s, nominal=%.6f Hz, effective=%.6f Hz); no resampling.",
        processing_fs,
        config.sampling_frequency_mode,
        config.nominal_fs,
        recording.effective_fs,
    )

    # channel_separation: split EEG and EOG; EOG is never used for interpolation/reference.
    groups = separate_channel_types(recording.ch_names, config)
    if config.eog_required and not groups.eog_indices:
        raise PreprocessingError(
            "No EOG channel was found while eog_required=True; recording requires manual review."
        )

    # mne_raw_creation: attach channel types and standard_1020 montage by default.
    raw, unmatched_montage_channels, montage_warnings = create_raw(
        recording,
        groups,
        config,
        processing_fs,
    )
    raw_before = recording.data.copy()

    # constant_detrending: remove channel offsets before bad-channel checks.
    data = raw.get_data()
    all_signal_indices = list(range(data.shape[0]))
    data = detrend_channels(data, all_signal_indices, config)
    raw._data = data

    # flat_channel_detection: identify bad EEG channels and interpolate within guard limits.
    bad_channel_reasons = detect_bad_channels(
        raw.get_data(),
        groups,
        config,
        processing_fs,
    )
    raw, interpolated_channels, interpolation_warnings = interpolate_bad_channels(
        raw,
        bad_reasons=bad_channel_reasons,
        groups=groups,
        config=config,
    )

    # power_line_noise_removal: use ZapLine at 50 Hz; fallback notch only if needed.
    data, zapline_report = apply_zapline(
        raw.get_data(),
        groups,
        config,
        processing_fs,
    )

    # filtering: zero-phase Butterworth band-pass for EEG and low-pass for EOG.
    data = filter_eeg(data, groups, config, processing_fs)
    data = filter_eog(data, groups, config, processing_fs)

    # sixty_second_crop: crop the middle 60 seconds after filtering.
    crop = crop_recording(data, recording.timestamps, config, processing_fs)

    # average_reference: common-average reference over EEG channels only.
    referenced = apply_average_reference(crop.data, groups)

    # epoching: fixed 4-second epochs, with aligned EOG epochs if present.
    epochs = create_fixed_epochs(
        referenced,
        crop.timestamps,
        groups,
        config,
        processing_fs,
    )

    # bad_epoch_rejection: reject only by documented hybrid quality criteria.
    rejection = reject_bad_epochs(epochs.eeg_epochs, groups.eeg_ch_names, config)

    # output_metadata: collect every meaningful warning before assigning quality_status.
    filter_parameters = make_filter_parameters(config, zapline_report, processing_fs)
    warning_messages = list(validation.warnings)
    warning_messages.extend(interpolation_warnings)
    warning_messages.extend(montage_warnings)
    if unmatched_montage_channels:
        warning_messages.append(
            "Montage did not match EEG channels: "
            + ", ".join(unmatched_montage_channels)
        )
    for channel_type in ("eeg", "eog"):
        report = zapline_report.get(channel_type, {})
        if report.get("warning"):
            warning_messages.append(str(report["warning"]))
        if report.get("zapline_error") and report.get("fallback_notch_harmonics"):
            warning_messages.append(
                f"ZapLine failed for {channel_type.upper()}; notch fallback was applied "
                f"at {report['fallback_notch_harmonics']} Hz."
            )

    # qc_plots: use all pre-rejection EEG epochs so rejected traces remain available.
    plot_paths, plot_warnings = generate_qc_plots(
        output_dir=output_dir,
        input_path=input_path,
        raw_before=raw_before,
        processed_crop=referenced,
        groups=groups,
        config=config,
        processing_fs=processing_fs,
        all_eeg_epochs=epochs.eeg_epochs,
        eeg_ch_names=groups.eeg_ch_names,
        rejected_epoch_indices=rejection.rejected_indices,
        rejection_reasons=rejection.rejection_reasons,
        bad_channel_reasons=bad_channel_reasons,
    )
    warning_messages.extend(plot_warnings)
    warning_messages = list(dict.fromkeys(warning_messages))
    quality_status = "OK_WITH_WARNINGS" if warning_messages else "OK"

    # save_output: compressed NPZ contains retained EEG epochs and aligned metadata.
    output_path = save_processed_recording(
        output_dir=output_dir,
        input_path=input_path,
        epochs=epochs,
        rejection=rejection,
        groups=groups,
        crop=crop,
        original_duration=recording.original_duration,
        interpolated_channels=interpolated_channels,
        filter_parameters=filter_parameters,
        quality_status=quality_status,
        config=config,
        processing_fs=processing_fs,
    )

    # qc_report: JSON report preserves validation, bad channels and epoch reasons.
    qc_report_path = generate_qc_report(
        output_dir=output_dir,
        input_path=input_path,
        validation=validation,
        groups=groups,
        crop=crop,
        interpolated_channels=interpolated_channels,
        bad_channel_reasons=bad_channel_reasons,
        rejection=rejection,
        output_path=output_path,
        quality_status=quality_status,
        config=config,
        processing_fs=processing_fs,
        warning_messages=warning_messages,
        unmatched_montage_channels=unmatched_montage_channels,
        zapline_report=zapline_report,
        plot_paths=plot_paths,
        extra={
            "interpolation_warnings": interpolation_warnings,
            "filter_parameters": filter_parameters,
        },
    )

    return ProcessingSummary(
        input_path=input_path,
        output_path=output_path,
        qc_report_path=qc_report_path,
        original_duration=recording.original_duration,
        cropped_duration=crop.requested_duration,
        n_eeg_channels=len(groups.eeg_ch_names),
        n_eog_channels=len(groups.eog_ch_names),
        interpolated_channels=interpolated_channels,
        total_epochs=int(epochs.eeg_epochs.shape[0]),
        rejected_epochs=len(rejection.rejected_indices),
        retained_epochs=len(rejection.retained_indices),
        quality_status=quality_status,
    )


def write_failure_report(
    input_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    error: Exception,
) -> Path:
    """Write a minimal QC report when processing cannot continue."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{input_path.stem}_qc_failed.json"
    report = {
        "input_path": str(input_path),
        "quality_status": "FAILED_OR_REQUIRES_MANUAL_REVIEW",
        "error": str(error),
        "config": config,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(report), handle, indent=2, sort_keys=True)
    return report_path


def iter_input_files(
    input_path: Path,
    batch: bool,
    config: PreprocessingConfig | None = None,
) -> list[Path]:
    """Resolve one input or case-insensitively find configured batch suffixes."""
    config = config or PreprocessingConfig()
    validate_config(config)
    extensions = {extension.lower() for extension in config.input_file_extensions}
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir() and batch:
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
    if input_path.is_dir():
        supported = ", ".join(sorted(extensions))
        raise ValueError(
            "Input path is a directory; pass --batch to process supported files "
            f"({supported})."
        )
    raise FileNotFoundError(input_path)


def print_summary(summary: ProcessingSummary) -> None:
    """Print the concise end-of-file processing summary."""
    print(f"Input: {summary.input_path}")
    print(f"Quality status: {summary.quality_status}")
    print(f"Original duration: {summary.original_duration:.6f} s")
    print(f"Cropped duration: {summary.cropped_duration:.6f} s")
    print(f"EEG channels: {summary.n_eeg_channels}")
    print(f"EOG channels: {summary.n_eog_channels}")
    print(
        "Interpolated channels: "
        + (", ".join(summary.interpolated_channels) if summary.interpolated_channels else "none")
    )
    print(f"Total generated epochs: {summary.total_epochs}")
    print(f"Rejected epochs: {summary.rejected_epochs}")
    print(f"Retained epochs: {summary.retained_epochs}")
    print(f"Output path: {summary.output_path if summary.output_path else 'not saved'}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess continuous baseline EEG CSV recordings."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="CSV file to process, or a directory when --batch is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_baselines"),
        help="Directory for NPZ outputs and QC JSON reports.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file overriding PreprocessingConfig fields.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate only; print proposed operations and do not save.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug logging and process only one file if input_path is a directory.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process every compatible CSV file in input_path when it is a directory.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate optional non-blocking QC plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    config = load_config(args.config)
    if args.plots:
        config.generate_plots = True

    input_files = iter_input_files(
        args.input_path,
        batch=args.batch or args.debug,
        config=config,
    )
    if args.debug:
        input_files = input_files[:1]

    summaries: list[ProcessingSummary] = []
    for input_file in input_files:
        try:
            summary = process_recording(
                input_path=input_file,
                output_dir=args.output_dir,
                config=config,
                dry_run=args.dry_run,
            )
            summaries.append(summary)
            print_summary(summary)
        except Exception as exc:  # noqa: BLE001 - CLI should continue in batch mode.
            LOGGER.exception("Failed to process %s", input_file)
            report_path = None if args.dry_run else write_failure_report(input_file, args.output_dir, config, exc)
            failed = ProcessingSummary(
                input_path=input_file,
                output_path=None,
                qc_report_path=report_path,
                original_duration=float("nan"),
                cropped_duration=0.0,
                n_eeg_channels=0,
                n_eog_channels=0,
                interpolated_channels=[],
                total_epochs=0,
                rejected_epochs=0,
                retained_epochs=0,
                quality_status="FAILED_OR_REQUIRES_MANUAL_REVIEW",
            )
            summaries.append(failed)
            print_summary(failed)
            if not args.batch and not args.debug:
                raise

    if len(summaries) > 1:
        retained_total = sum(summary.retained_epochs for summary in summaries)
        failed_total = sum(
            summary.quality_status == "FAILED_OR_REQUIRES_MANUAL_REVIEW"
            for summary in summaries
        )
        print(
            f"Batch summary: files={len(summaries)}, failed/manual_review={failed_total}, "
            f"retained_epochs={retained_total}, output_dir={args.output_dir}"
        )


if __name__ == "__main__":
    main()

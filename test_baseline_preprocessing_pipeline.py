from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_preprocessing_pipeline as pipeline


HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_MNE = importlib.util.find_spec("mne") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAS_MEEGKIT = importlib.util.find_spec("meegkit") is not None


def make_synthetic_csv(path: Path, *, include_eog: bool = True) -> None:
    """Write a deterministic approximately 65-second, 1024-Hz baseline CSV."""
    fs = 1024.0
    n_samples = int(65 * fs)
    time = np.arange(n_samples, dtype=float) / fs
    eeg_names = ["Fp1", "Fp2", "F3", "F4", "Cz", "Pz", "O1", "O2"]

    columns: dict[str, np.ndarray] = {"stream_time": time}
    for channel_idx, name in enumerate(eeg_names):
        phase = channel_idx * 0.2
        signal_volts = (
            20e-6 * np.sin(2 * np.pi * 10 * time + phase)
            + 5e-6 * np.sin(2 * np.pi * 50 * time)
        )
        columns[name] = signal_volts * 1e3  # CSV input unit is mV.

    columns["O2"] = np.full(n_samples, 0.01, dtype=float)  # Flat EEG in mV.
    contamination = (time >= 30.0) & (time < 30.05)
    columns["Fp1"][contamination] += 2.0  # 2 mV excursion in one analysis epoch.
    if include_eog:
        columns["EOG"] = 0.1 * np.sin(2 * np.pi * time)
    columns["Trigger"] = np.zeros(n_samples, dtype=int)
    columns["Sample Counter"] = np.arange(n_samples, dtype=int)
    pd.DataFrame(columns).to_csv(path, sep=";", index=False)


def make_epoch_fixture(has_eog: bool = True) -> tuple[
    pipeline.EpochResult,
    pipeline.ChannelGroups,
    pipeline.CropResult,
    pipeline.RejectionResult,
]:
    n_epochs = 2
    n_samples = 4096
    eeg_epochs = np.zeros((n_epochs, 2, n_samples), dtype=float)
    eog_epochs = (
        np.zeros((n_epochs, 1, n_samples), dtype=float) if has_eog else None
    )
    epochs = pipeline.EpochResult(
        eeg_epochs=eeg_epochs,
        eog_epochs=eog_epochs,
        epoch_start_times=np.asarray([0.0, 4.0]),
        epoch_end_times=np.asarray([4.0, 8.0]),
        samples_per_epoch=n_samples,
    )
    groups = pipeline.ChannelGroups(
        eeg_indices=[0, 1],
        eog_indices=[2] if has_eog else [],
        eeg_ch_names=["Fp1", "Fp2"],
        eog_ch_names=["EOG"] if has_eog else [],
    )
    timestamps = np.arange(60 * 1024, dtype=float) / 1024.0
    crop = pipeline.CropResult(
        data=np.zeros((3 if has_eog else 2, timestamps.size), dtype=float),
        timestamps=timestamps,
        crop_start=0.0,
        crop_end=60.0,
        requested_duration=60.0,
        actual_timestamp_span=float(timestamps[-1] - timestamps[0]),
        n_samples=timestamps.size,
        expected_samples=timestamps.size,
        sample_coverage_duration=60.0,
        first_to_last_timestamp_span=float(timestamps[-1] - timestamps[0]),
    )
    rejection = pipeline.RejectionResult(
        retained_indices=[0, 1],
        rejected_indices=[],
        rejection_reasons={},
        epoch_quality_details={0: {}, 1: {}},
    )
    return epochs, groups, crop, rejection


def make_batch_summary(filename: str, session: int) -> pipeline.ProcessingSummary:
    epoch_metrics = [
        {
            "original_epoch_index": epoch_idx,
            "median_line_noise_ratio_db": 4.0 + session,
            "maximum_line_noise_ratio_db": 7.0 + session,
            "median_peak_to_peak_microvolts": 80.0 + session,
            "maximum_peak_to_peak_microvolts": 140.0 + session,
            "high_line_noise_channel_count": 1,
        }
        for epoch_idx in range(15)
    ]
    return pipeline.ProcessingSummary(
        input_path=Path(filename),
        output_path=None,
        qc_report_path=None,
        original_duration=65.0,
        cropped_duration=60.0,
        n_eeg_channels=8,
        n_eog_channels=1,
        interpolated_channels=["O2"],
        total_epochs=15,
        rejected_epochs=1,
        retained_epochs=14,
        quality_status="OK_WITH_WARNINGS",
        effective_fs=1024.0,
        retention_fraction=14 / 15,
        manual_review_required=False,
        raw_noise_metrics={
            "recording_summary": {
                "median_line_noise_ratio_db": 4.0 + session,
                "maximum_line_noise_ratio_db": 7.0 + session,
                "high_line_noise_channel_fraction": 0.125,
                "median_peak_to_peak_microvolts": 80.0 + session,
                "maximum_peak_to_peak_microvolts": 140.0 + session,
            },
            "epoch_metrics": epoch_metrics,
        },
        rejection_reasons={2: ["extreme_peak_to_peak: Fp1"]},
    )


class PipelineRegressionTests(unittest.TestCase):
    def test_zero_retained_epochs_is_recording_failure(self) -> None:
        status, fraction, manual_review, reasons = (
            pipeline.determine_recording_quality_status(
                15,
                0,
                [],
                pipeline.PreprocessingConfig(),
            )
        )
        self.assertEqual(status, "FAILED_OR_REQUIRES_MANUAL_REVIEW")
        self.assertEqual(fraction, 0.0)
        self.assertTrue(manual_review)
        self.assertIn("zero_retained_epochs", reasons)

    def test_six_of_fifteen_retained_epochs_warns(self) -> None:
        status, fraction, manual_review, reasons = (
            pipeline.determine_recording_quality_status(
                15,
                6,
                [],
                pipeline.PreprocessingConfig(),
            )
        )
        self.assertEqual(status, "OK_WITH_WARNINGS")
        self.assertAlmostEqual(fraction, 0.4)
        self.assertTrue(manual_review)
        self.assertIn("retained_epochs_below_minimum: 6 < 10", reasons)
        self.assertIn("retention_fraction_below_minimum: 0.4 < 0.67", reasons)

    def test_fifteen_retained_epochs_can_remain_ok(self) -> None:
        status, fraction, manual_review, reasons = (
            pipeline.determine_recording_quality_status(
                15,
                15,
                [],
                pipeline.PreprocessingConfig(),
            )
        )
        self.assertEqual(status, "OK")
        self.assertEqual(fraction, 1.0)
        self.assertFalse(manual_review)
        self.assertEqual(reasons, [])

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for filtering tests")
    def test_sos_filter_and_eeg_eog_filters_preserve_shape(self) -> None:
        from scipy import signal

        config = pipeline.PreprocessingConfig(zapline_enabled=False)
        data = np.random.default_rng(4).normal(size=(3, 4096))
        sos = signal.butter(4, [0.5, 80.0], btype="bandpass", fs=1024, output="sos")
        filtered = pipeline.sosfiltfilt_checked(data, sos, config, "test filter")
        self.assertEqual(filtered.shape, data.shape)

        groups = pipeline.ChannelGroups(
            eeg_indices=[0, 1],
            eog_indices=[2],
            eeg_ch_names=["Fp1", "Fp2"],
            eog_ch_names=["EOG"],
        )
        eeg_filtered = pipeline.filter_eeg(data, groups, config, 1024.0)
        eog_filtered = pipeline.filter_eog(eeg_filtered, groups, config, 1024.0)
        self.assertEqual(eeg_filtered.shape, data.shape)
        self.assertEqual(eog_filtered.shape, data.shape)

    def test_eog_zapline_disabled_skips_dss_and_fallback_notch(self) -> None:
        fs = 1024.0
        time = np.arange(4096, dtype=float) / fs
        data = np.sin(2 * np.pi * 50 * time)[np.newaxis, :]
        groups = pipeline.ChannelGroups(
            eeg_indices=[],
            eog_indices=[0],
            eeg_ch_names=[],
            eog_ch_names=["EOG"],
        )
        config = pipeline.PreprocessingConfig(
            zapline_enabled=True,
            zapline_eog_enabled=False,
            fallback_notch_enabled=True,
        )

        output, report = pipeline.apply_zapline(data, groups, config, fs)

        np.testing.assert_array_equal(output, data)
        self.assertTrue(report["eog"]["processing_skipped"])
        self.assertEqual(report["eog"]["skip_reason"], "disabled_by_configuration")
        self.assertEqual(report["eog"]["fallback_notch_harmonics"], [])

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for EOG low-pass testing")
    def test_single_eog_skips_zapline_then_receives_only_lowpass(self) -> None:
        fs = 1024.0
        time = np.arange(4096, dtype=float) / fs
        data = np.stack(
            [
                20e-6 * np.sin(2 * np.pi * 10 * time),
                15e-6 * np.sin(2 * np.pi * 12 * time),
                50e-6 * np.sin(2 * np.pi * 2 * time)
                + 100e-6 * np.sin(2 * np.pi * 50 * time),
            ]
        )
        groups = pipeline.ChannelGroups(
            eeg_indices=[0, 1],
            eog_indices=[2],
            eeg_ch_names=["Fp1", "Fp2"],
            eog_ch_names=["EOG"],
        )
        config = pipeline.PreprocessingConfig(
            zapline_enabled=False,
            zapline_eog_enabled=True,
            fallback_notch_enabled=False,
        )

        with self.assertLogs("baseline_preprocessing", level="WARNING") as captured:
            after_zapline, report = pipeline.apply_zapline(data, groups, config, fs)

        np.testing.assert_array_equal(after_zapline[groups.eog_indices], data[groups.eog_indices])
        self.assertEqual(report["eog"]["skip_reason"], "single_eog_channel")
        self.assertTrue(any("single EOG channel" in line for line in captured.output))

        after_lowpass = pipeline.filter_eog(after_zapline, groups, config, fs)
        np.testing.assert_array_equal(
            after_lowpass[groups.eeg_indices],
            after_zapline[groups.eeg_indices],
        )
        self.assertEqual(after_lowpass.shape, data.shape)
        self.assertGreater(
            float(np.max(np.abs(after_lowpass[2] - after_zapline[2]))),
            10e-6,
        )

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for raw noise metrics")
    def test_50hz_sinusoid_has_higher_raw_line_ratio(self) -> None:
        fs = 1024.0
        time = np.arange(int(4 * fs), dtype=float) / fs
        random = np.random.default_rng(22)
        background = 8e-6 * np.sin(2 * np.pi * 10 * time)
        noise = random.normal(scale=1e-6, size=time.size)
        clean = background + noise
        contaminated = clean + 20e-6 * np.sin(2 * np.pi * 50 * time)
        metrics = pipeline.calculate_raw_noise_metrics(
            np.stack([clean, contaminated]),
            ["Clean", "Line50"],
            fs,
            pipeline.PreprocessingConfig(),
        )
        channel_metrics = metrics["channel_metrics"]
        self.assertGreater(
            channel_metrics["Line50"]["line_noise_ratio_db"],
            channel_metrics["Clean"]["line_noise_ratio_db"] + 10.0,
        )

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for raw noise metrics")
    def test_raw_noise_metrics_do_not_modify_input_and_create_fifteen_epochs(self) -> None:
        fs = 1024.0
        time = np.arange(int(60 * fs), dtype=float) / fs
        data = np.stack(
            [
                10e-6 * np.sin(2 * np.pi * 10 * time),
                10e-6 * np.sin(2 * np.pi * 10 * time)
                + 4e-6 * np.sin(2 * np.pi * 50 * time),
            ]
        )
        original = data.copy()
        metrics = pipeline.calculate_raw_noise_metrics(
            data,
            ["Fp1", "Fp2"],
            fs,
            pipeline.PreprocessingConfig(),
        )
        np.testing.assert_array_equal(data, original)
        self.assertEqual(len(metrics["epoch_metrics"]), 15)
        self.assertNotIn("data", metrics)
        self.assertNotIn("psd", metrics)

    @unittest.skipUnless(HAS_SCIPY, "SciPy is required for notch comparison")
    def test_raw_metrics_are_calculated_before_notch_filtering(self) -> None:
        fs = 1024.0
        time = np.arange(int(8 * fs), dtype=float) / fs
        data = (
            10e-6 * np.sin(2 * np.pi * 10 * time)
            + 30e-6 * np.sin(2 * np.pi * 50 * time)
        )[np.newaxis, :]
        config = pipeline.PreprocessingConfig()
        raw_metrics = pipeline.calculate_raw_noise_metrics(data, ["Fp1"], fs, config)
        notched = pipeline.notch_filter_array(data, fs, 50.0, config)
        post_notch_metrics = pipeline.calculate_raw_noise_metrics(
            notched,
            ["Fp1"],
            fs,
            config,
        )
        self.assertGreater(
            raw_metrics["recording_summary"]["median_line_noise_ratio_db"],
            post_notch_metrics["recording_summary"]["median_line_noise_ratio_db"],
        )

    def test_crop_uses_sample_coverage_not_first_last_span(self) -> None:
        fs = 1024.0
        timestamps = np.arange(int(65 * fs), dtype=float) / fs
        data = np.zeros((2, timestamps.size), dtype=float)
        crop = pipeline.crop_recording(
            data,
            timestamps,
            pipeline.PreprocessingConfig(),
            processing_fs=fs,
        )
        self.assertEqual(crop.n_samples, 60 * 1024)
        self.assertEqual(crop.expected_samples, 60 * 1024)
        self.assertAlmostEqual(crop.sample_coverage_duration, 60.0, places=9)
        self.assertAlmostEqual(
            crop.first_to_last_timestamp_span,
            60.0 - 1.0 / fs,
            places=9,
        )

    def test_four_second_epochs_produce_fifteen(self) -> None:
        fs = 1024.0
        samples = 60 * 1024
        timestamps = np.arange(samples, dtype=float) / fs
        data = np.zeros((3, samples), dtype=float)
        groups = pipeline.ChannelGroups(
            eeg_indices=[0, 1],
            eog_indices=[2],
            eeg_ch_names=["Fp1", "Fp2"],
            eog_ch_names=["EOG"],
        )
        epochs = pipeline.create_fixed_epochs(
            data,
            timestamps,
            groups,
            pipeline.PreprocessingConfig(),
            processing_fs=fs,
        )
        self.assertEqual(epochs.eeg_epochs.shape, (15, 2, 4096))
        self.assertEqual(epochs.eog_epochs.shape, (15, 1, 4096))

    def test_rejection_records_every_specific_cause_deterministically(self) -> None:
        config = pipeline.PreprocessingConfig(max_bad_channels=1)
        time = np.arange(4096, dtype=float) / 1024.0
        epochs = np.stack(
            [10e-6 * np.sin(2 * np.pi * (8 + idx) * time) for idx in range(5)]
        )[np.newaxis, ...]
        epochs[0, 4] = 0.0
        epochs[0, 0, 0] = np.nan
        epochs[0, 1, 100] = 400e-6
        epochs[0, 2, 100] = 1200e-6
        epochs[0, 3, 100] = -500e-6
        ch_names = ["Fp1", "Fp2", "F3", "F4", "Cz"]

        result = pipeline.reject_bad_epochs(epochs, ch_names, config)
        reasons = result.rejection_reasons[0]
        self.assertEqual(result.rejected_indices, [0])
        self.assertTrue(any(reason == "invalid_values: Fp1" for reason in reasons))
        self.assertTrue(any(reason.startswith("high_peak_to_peak:") for reason in reasons))
        self.assertTrue(any(reason == "extreme_peak_to_peak: F3" for reason in reasons))
        self.assertTrue(any(reason.startswith("large_absolute_excursion:") for reason in reasons))
        self.assertTrue(any(reason == "flat_signal: Cz" for reason in reasons))
        self.assertTrue(any(reason.startswith("too_many_contaminated_channels:") for reason in reasons))
        self.assertTrue(result.epoch_quality_details[0]["contaminated_channel_limit_exceeded"])

    def test_qc_json_is_compact_and_excludes_signal_arrays(self) -> None:
        epochs, groups, crop, rejection = make_epoch_fixture()
        validation = pipeline.ValidationReport(3, crop.n_samples, 65.0, 1024.0, [])
        config = pipeline.PreprocessingConfig()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = pipeline.generate_qc_report(
                Path(temp_dir),
                Path("recording.csv"),
                validation,
                groups,
                crop,
                [],
                {},
                rejection,
                None,
                "OK",
                config,
                1024.0,
                zapline_report={
                    "eeg": {},
                    "eog": {
                        "processing_skipped": True,
                        "skip_reason": "disabled_by_configuration",
                    },
                },
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn("data", report["crop"])
            self.assertNotIn("timestamps", report["crop"])
            self.assertTrue(report["eog_line_noise_processing"]["processing_skipped"])
            self.assertEqual(
                report["eog_line_noise_processing"]["skip_reason"],
                "disabled_by_configuration",
            )
            self.assertLess(report_path.stat().st_size, 100_000)

    def test_npz_without_eog_is_pickle_free(self) -> None:
        epochs, groups, crop, rejection = make_epoch_fixture(has_eog=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pipeline.save_processed_recording(
                Path(temp_dir),
                Path("recording.csv"),
                epochs,
                rejection,
                groups,
                crop,
                65.0,
                [],
                {},
                "OK",
                pipeline.PreprocessingConfig(),
                1024.0,
            )
            with np.load(output, allow_pickle=False) as saved:
                self.assertFalse(bool(saved["has_eog"]))
                self.assertEqual(saved["eog_data"].shape, (0, 0, 0))
                self.assertTrue(all(saved[name].dtype != object for name in saved.files))

    def test_unsupported_input_format_has_clear_error(self) -> None:
        with self.assertRaisesRegex(pipeline.PreprocessingError, r"currently: \.csv"):
            pipeline.load_baseline_recording(
                Path("recording.edf"),
                pipeline.PreprocessingConfig(),
            )

    def test_batch_discovery_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            lower = directory / "one.csv"
            upper = directory / "two.CSV"
            ignored = directory / "three.txt"
            lower.touch()
            upper.touch()
            ignored.touch()
            found = pipeline.iter_input_files(
                directory,
                batch=True,
                config=pipeline.PreprocessingConfig(),
            )
            self.assertEqual(found, [lower, upper])

    def test_filename_parsing(self) -> None:
        parsed = pipeline.parse_baseline_filename(
            "sub-s01_ses-6_run-001_bl-01.csv"
        )
        self.assertEqual(
            parsed,
            {
                "subject": "s01",
                "session": "6",
                "run": "001",
                "baseline_number": "01",
            },
        )

    def test_batch_csv_row_counts_match_recordings_and_original_epochs(self) -> None:
        summaries = [
            make_batch_summary("sub-s01_ses-1_run-001_bl-01.csv", 1),
            make_batch_summary("sub-s02_ses-2_run-001_bl-01.csv", 2),
        ]
        recording_rows: list[dict[str, object]] = []
        epoch_rows: list[dict[str, object]] = []
        for summary in summaries:
            recording_row, recording_epoch_rows = pipeline.build_batch_qc_rows(summary)
            recording_rows.append(recording_row)
            epoch_rows.extend(recording_epoch_rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            recording_path, epoch_path = pipeline.write_batch_summary_csvs(
                Path(temp_dir),
                recording_rows,
                epoch_rows,
            )
            recording_frame = pd.read_csv(recording_path)
            epoch_frame = pd.read_csv(epoch_path)
            self.assertEqual(len(recording_frame), 2)
            self.assertEqual(len(epoch_frame), 30)
            self.assertEqual(
                int((epoch_frame["epoch_status"] == "rejected").sum()),
                2,
            )

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for batch plots")
    def test_batch_plots_are_generated_without_interactive_windows(self) -> None:
        import matplotlib.pyplot as pyplot

        summaries = [
            make_batch_summary("sub-s01_ses-1_run-001_bl-01.csv", 1),
            make_batch_summary("sub-s01_ses-2_run-001_bl-01.csv", 2),
            make_batch_summary("sub-s02_ses-1_run-001_bl-01.csv", 1),
            make_batch_summary("sub-s02_ses-2_run-001_bl-01.csv", 2),
        ]
        rows = [pipeline.build_batch_qc_rows(summary)[0] for summary in summaries]
        with tempfile.TemporaryDirectory() as temp_dir:
            saved, plot_warnings = pipeline.generate_batch_qc_plots(
                Path(temp_dir),
                rows,
            )
            self.assertEqual(plot_warnings, [])
            self.assertEqual(
                {path.name for path in saved},
                {
                    "batch_raw_50hz_ratio_by_session.png",
                    "batch_raw_peak_to_peak_by_session.png",
                    "batch_epoch_retention_by_session.png",
                },
            )
            self.assertEqual(pyplot.get_fignums(), [])

    @unittest.skipUnless(
        HAS_SCIPY and HAS_MNE and HAS_MATPLOTLIB,
        "SciPy, MNE, and matplotlib are required for the full integration test",
    )
    def test_complete_synthetic_pipeline_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            input_path = directory / "participant_01.csv"
            output_dir = directory / "output"
            make_synthetic_csv(input_path)
            config = pipeline.PreprocessingConfig(
                amplitude_unit="mV",
                zapline_enabled=False,
                fallback_notch_enabled=True,
                generate_plots=True,
                max_bad_channels=1,
                max_rejected_epoch_plots=2,
            )
            summary = pipeline.process_recording(input_path, output_dir, config)

            self.assertEqual(summary.total_epochs, 15)
            self.assertGreaterEqual(summary.rejected_epochs, 1)
            self.assertEqual(summary.quality_status, "OK_WITH_WARNINGS")
            with np.load(summary.output_path, allow_pickle=False) as saved:
                self.assertEqual(saved["data"].shape[1:], (8, 4096))
                self.assertEqual(saved["eog_data"].shape[0], saved["data"].shape[0])
                self.assertTrue(bool(saved["has_eog"]))
                self.assertTrue(all(saved[name].dtype != object for name in saved.files))

            qc = json.loads(summary.qc_report_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["crop"]["expected_samples"], 60 * 1024)
            self.assertEqual(qc["total_generated_epochs"], 15)
            self.assertAlmostEqual(
                qc["retention_fraction"],
                summary.retained_epochs / 15,
            )
            self.assertTrue(qc["rejection_reasons"])
            self.assertEqual(len(qc["raw_noise_metrics"]["epoch_metrics"]), 15)
            self.assertNotIn("data", qc["raw_noise_metrics"])
            self.assertNotIn("psd", qc["raw_noise_metrics"])
            self.assertTrue(qc["warning_messages"])
            self.assertLess(summary.qc_report_path.stat().st_size, 250_000)
            plot_names = {path.name for path in (output_dir / "plots").glob("*.png")}
            self.assertIn("participant_01_traces.png", plot_names)
            self.assertIn("participant_01_psd.png", plot_names)
            self.assertTrue(
                any(name.startswith("participant_01_rejected_epoch_") for name in plot_names)
            )

    @unittest.skipUnless(HAS_MEEGKIT, "meegkit is required for the ZapLine shape test")
    def test_meegkit_zapline_preserves_expected_dimensions(self) -> None:
        fs = 1024.0
        time = np.arange(4096, dtype=float) / fs
        data = np.stack(
            [
                np.sin(2 * np.pi * 10 * time) + 0.2 * np.sin(2 * np.pi * 50 * time),
                np.sin(2 * np.pi * 12 * time) + 0.2 * np.sin(2 * np.pi * 50 * time),
                np.sin(2 * np.pi * 8 * time) + 0.2 * np.sin(2 * np.pi * 50 * time),
            ]
        )
        config = pipeline.PreprocessingConfig(
            zapline_enabled=True,
            zap_harmonics=(1,),
            zap_nremove_eeg=1,
            fallback_notch_enabled=False,
        )
        output, report = pipeline.apply_zapline_to_indices(
            data,
            [0, 1, 2],
            fs,
            1,
            config,
            "EEG",
        )
        self.assertEqual(output.shape, data.shape)
        self.assertEqual(report["processed_harmonics"], [50.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

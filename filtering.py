import numpy as np
from scipy.signal import butter, filtfilt, detrend
from meegkit.dss import dss_line

# ================= CONFIGURATION =================
# Epochs recorded/stored at 1024 Hz, 5s -> 5120 samples
FS = 1024
LOWCUT = 0.5
HIGHCUT = 80.0
ORDER = 4
NOTCH_FREQ = 50.0

# Late slicing window bounds inside each 5s epoch
CUT_START = 1741
CUT_END = 3789

# ---- Pre-ICA per-epoch hybrid gating ----
# Flags/drops obviously-bad epochs (huge amplitude excursions, too many
# bad channels at once) BEFORE ICA is trained, so ICA doesn't fit noise.
PRE_ICA_VOLT_THRESH = 350e-6
PRE_ICA_MAX_BAD_CHS = 5

# ---- Bad/flat channel interpolation cap ----
# Interpolate freely up to ~10-15% of channels, treat 15-20% as a soft
# ceiling worth flagging, and above ~20% prefer dropping the channel
# outright rather than trusting the interpolation.
MAX_INTERP_FRACTION_SOFT = 0.15
MAX_INTERP_FRACTION_HARD = 0.20

# ZapLine configuration
ZAP_HARMONICS = [1, 2, 3]
ZAP_NREMOVE_EEG = 5
ZAP_NREMOVE_EOG = 1


# ================= HELPERS =================

def interpolate_or_drop_channels(epochs_mne, dead_chs, eeg_ch_names):
    """
    Apply the interpolation cap: interpolate freely under the soft
    threshold, still interpolate (with a warning) up to the hard
    threshold, and drop channels outright beyond that rather than trust
    the interpolation.
    Returns (epochs_mne, updated_ch_names, interpolated, dropped).
    """
    n_ch = len(eeg_ch_names)
    frac = len(dead_chs) / n_ch if n_ch else 0.0
    interpolated, dropped = [], []

    if not dead_chs:
        return epochs_mne, eeg_ch_names, interpolated, dropped

    if frac <= MAX_INTERP_FRACTION_HARD:
        if frac > MAX_INTERP_FRACTION_SOFT:
            print(f"  [Interp Guard] {frac * 100:.1f}% of channels flat/dead "
                  f"(> {MAX_INTERP_FRACTION_SOFT * 100:.0f}% soft cap, "
                  f"<= {MAX_INTERP_FRACTION_HARD * 100:.0f}% hard cap). Interpolating, but flagging for review.")
        epochs_mne.info['bads'] = dead_chs
        epochs_mne.interpolate_bads(reset_bads=True, verbose=False)
        interpolated = list(dead_chs)
    else:
        print(f"  [Interp Guard] {frac * 100:.1f}% of channels flat/dead exceeds "
              f"{MAX_INTERP_FRACTION_HARD * 100:.0f}% hard cap. Dropping channels instead of interpolating.")
        epochs_mne.drop_channels(dead_chs)
        eeg_ch_names = [ch for ch in eeg_ch_names if ch not in dead_chs]
        dropped = list(dead_chs)

    return epochs_mne, eeg_ch_names, interpolated, dropped


# ================= PRE-ICA PROCESSING (per run) =================
# eeg_data: (n_epochs, n_channels, n_samples), already split from raw .npz
# eog_data: same shape convention, or None if no EOG channels

# ---- Detrend ----
eeg_data = detrend(eeg_data, axis=-1, type='constant')
if eog_data is not None:
    eog_data = detrend(eog_data, axis=-1, type='constant')

# ---- Dead/flat channel detection + capped interpolation ----
epochs_temp = pp.create_mne_epochs(eeg_data, eeg_ch_names, FS)
try:
    epochs_temp.set_montage('standard_1020', on_missing='ignore')
except Exception:
    pass
dead_chs = pp.find_flat_channels(epochs_temp)

interpolated_chs, dropped_chs = [], []
if dead_chs:
    epochs_temp, eeg_ch_names, interpolated_chs, dropped_chs = interpolate_or_drop_channels(
        epochs_temp, dead_chs, eeg_ch_names)
    eeg_data = epochs_temp.get_data()

# ---- ZapLine (line-noise removal) ----
eeg_zap = eeg_data.transpose(2, 1, 0)  # (samples, chs, epochs)
for h in ZAP_HARMONICS:
    eeg_zap, _ = dss_line(eeg_zap, fline=NOTCH_FREQ * h, sfreq=FS, nremove=ZAP_NREMOVE_EEG)
eeg_data = eeg_zap.transpose(2, 1, 0)

if eog_data is not None:
    eog_zap = eog_data.transpose(2, 1, 0)
    for h in ZAP_HARMONICS:
        eog_zap, _ = dss_line(eog_zap, fline=NOTCH_FREQ * h, sfreq=FS, nremove=ZAP_NREMOVE_EOG)
    eog_data = eog_zap.transpose(2, 1, 0)

# ---- Bandpass filter ----
b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], btype='band')
eeg_data = filtfilt(b, a, eeg_data, axis=-1, padlen=150)
if eog_data is not None:
    b_eog, a_eog = butter(ORDER, 10.0 / (FS / 2), btype='low')
    eog_data = filtfilt(b_eog, a_eog, eog_data, axis=-1, padlen=150)

# ---- Cut to analysis window ----
eeg_cut = eeg_data[:, :, CUT_START:CUT_END]
eog_cut = eog_data[:, :, CUT_START:CUT_END] if eog_data is not None else None
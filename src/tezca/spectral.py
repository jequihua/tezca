"""Equal-segment-weight Welch on declared regular eligible runs."""

import importlib.metadata

import numpy as np
from scipy.signal import get_window, welch

from ._owned import identity, owned
from .individual import _ColumnSignal, _observations, _result
from .support import select_support


def _overlap_square_sum(window, hop, weights):
    """Weighted i<j overlap term for segments within one contiguous run."""
    energy = float(window @ window)
    penalty = 0.0
    for lag in range(1, min(len(weights), (len(window) - 1) // hop + 1)):
        shift = lag * hop
        rho = float(window[:-shift] @ window[shift:]) / energy
        penalty += float(weights[:-lag] @ weights[lag:]) * rho*rho
    return penalty


class WelchSpec:
    def __init__(self, nperseg=None, *, recipe="research_explicit", overlap=None, nfft=None,
                 taper="hann", periodic=True, detrend="constant", scaling="density",
                 onesided=True, average="mean", support="all_runs", min_segments=1,
                 min_nperseg=64, max_nperseg=4096, length_divisor=8):
        if recipe not in {"research_explicit", "sensor_length"}:
            raise ValueError("recipe must be research_explicit or sensor_length")
        for name, value, minimum in (("nperseg", nperseg, 2), ("overlap", overlap, 0), ("nfft", nfft, 2)):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >={minimum}")
        if recipe == "research_explicit" and nperseg is None:
            raise ValueError("research requires explicit nperseg; no automatic fallback")
        if recipe == "sensor_length" and nperseg is not None:
            raise ValueError("sensor_length computes its length; explicit nperseg conflicts")
        if (type(min_segments) is not int or min_segments < 1 or type(periodic) is not bool
                or onesided is not True or average != "mean"):
            raise ValueError("positive integer min_segments, boolean periodic, one-sided mean required")
        if taper not in {"boxcar", "hann"} or detrend not in {False, "constant", "linear"} or detrend == 0 and detrend is not False:
            raise ValueError("unsupported taper/detrend")
        if scaling not in {"density", "spectrum"} or support not in {"all_runs", "longest_run"}:
            raise ValueError("unsupported scaling/support")
        if nperseg is not None and ((overlap is not None and overlap >= nperseg) or (nfft is not None and nfft < nperseg)):
            raise ValueError("overlap must be below length; FFT length must be >=length")
        for name, value in (("min_nperseg", min_nperseg), ("max_nperseg", max_nperseg), ("length_divisor", length_divisor)):
            if type(value) is not int or value < (2 if name != "length_divisor" else 1):
                raise ValueError(f"{name} must be a positive integer with length >=2")
        if min_nperseg > max_nperseg:
            raise ValueError("minimum segment length exceeds maximum")
        if recipe == "research_explicit" and (min_nperseg, max_nperseg, length_divisor) != (64, 4096, 8):
            raise ValueError("length-rule settings apply only to sensor_length")
        self._record = dict(nperseg=nperseg, recipe=recipe, overlap=overlap, nfft=nfft,
            taper=taper, periodic=periodic, detrend=detrend, scaling=scaling, onesided=onesided,
            average=average, support=support, min_segments=min_segments)
        if recipe == "sensor_length":
            self._record.update(min_nperseg=min_nperseg, max_nperseg=max_nperseg, length_divisor=length_divisor)

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def power_spectrum(data, *, variable, spec, observations=None):
    if not isinstance(spec, WelchSpec):
        raise ValueError("power_spectrum requires WelchSpec")
    observations = _observations(observations)
    signal = _ColumnSignal(data, variable)
    requested = spec.record
    if signal.time.mode != "regular":
        selection = select_support(signal, observations=observations, policy="irregular", estimator="point")
        return _result(signal, selection, observations, metric="spectral.welch_mean_v1", value=None,
            status="unsupported_time_geometry", reason="Welch requires declared regular runs",
            units=signal.metadata[variable]["unit"] + ("^2/Hz" if requested["scaling"] == "density" else "^2"),
            config=requested, extra=dict(requested=requested, resolved=None, raw_segments=0))
    selection = select_support(signal, observations=observations, policy=requested["support"])
    selected = set(selection.positions.tolist())
    n = len(selected)
    length = requested["nperseg"]
    if requested["recipe"] == "sensor_length":
        length = max(requested["min_nperseg"], min(requested["max_nperseg"], n // requested["length_divisor"]))
    overlap = length // 2 if requested["overlap"] is None else requested["overlap"]
    nfft = length if requested["nfft"] is None else requested["nfft"]
    if overlap >= length or nfft < length:
        raise ValueError("resolved overlap must be <length and FFT >=length")
    hop = length - overlap
    fs = 1e9 / signal.time.cadence.value
    window = get_window(requested["taper"], length, fftbins=requested["periodic"])
    energy = float(window @ window)
    if energy == 0 or float(window.sum()) == 0:
        raise ValueError("taper has zero energy or coherent gain")
    runs, rejected, segments, used = [], [], [], set()
    values = signal.values.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
    weighted_psd = None
    weighted_physical_psd = None
    frequency = None
    penalty = 0.0
    for run in selection.record["runs"]:
        start, stop = run["start_position"], run["stop_position"]
        if start not in selected:
            rejected.append(dict(**run, reason="not_selected_by_support"))
            continue
        starts = list(range(start, stop - length + 1, hop))
        if not starts:
            rejected.append(dict(**run, reason="shorter_than_requested_segment"))
            continue
        f, p = welch(values[start:stop], fs=fs, window=window, nperseg=length,
                     noverlap=overlap, nfft=nfft, detrend=requested["detrend"],
                     return_onesided=True, scaling=requested["scaling"], axis=-1, average="mean")
        frequency = f
        weighted_psd = p * len(starts) if weighted_psd is None else weighted_psd + p * len(starts)
        if nfft == length:
            physical_p = p
        else:
            _, physical_p = welch(values[start:stop], fs=fs, window=window, nperseg=length,
                noverlap=overlap, nfft=length, detrend=requested["detrend"],
                return_onesided=True, scaling=requested["scaling"], axis=-1, average="mean")
        weighted_physical_psd = (physical_p * len(starts) if weighted_physical_psd is None
                                 else weighted_physical_psd + physical_p * len(starts))
        for segment_start in starts:
            cells = list(range(segment_start, segment_start + length))
            segments.append(dict(start_position=segment_start, stop_position=segment_start+length,
                                 run_start_position=start))
            used.update(cells)
        # Equal weights; only overlapping windows from the same run contribute.
        k = len(starts)
        penalty += _overlap_square_sum(window, hop, np.ones(k))
        runs.append(dict(**run, raw_segments=k, used_stop_position=starts[-1]+length,
                         used_duration_seconds=(starts[-1]+length-start)/fs))
    count = len(segments)
    used_selection = select_support(signal, observations=observations, policy="all_runs", within=sorted(used))
    physical_grid = np.fft.rfftfreq(length, d=1/fs)
    display_grid = np.fft.rfftfreq(nfft, d=1/fs)
    lower, upper = physical_grid[1], min(physical_grid[-1], display_grid[-1])
    resolved = dict(requested, nperseg=length, overlap=overlap, hop=hop, nfft=nfft,
                    sampling_hz=fs, frequency_units="Hz", window_energy=energy,
                    window_sum=float(window.sum()), density_to_spectrum_factor=energy*fs/float(window.sum())**2,
                    physical_resolution_hz=fs/length, display_spacing_hz=fs/nfft,
                    physical_positive_frequencies_hz=physical_grid[1:], theoretical_nyquist_hz=fs/2,
                    physical_positive_range_hz=[physical_grid[1], physical_grid[-1]],
                    display_positive_range_hz=[display_grid[1], display_grid[-1]],
                    supported_positive_range_hz=[lower, upper] if count and lower <= upper else None,
                    segment_aggregation="equal_segment_weights", scipy_version=importlib.metadata.version("scipy"))
    extra = dict(requested=requested, resolved=resolved, selected_positions=selection.positions.tolist(),
                 candidate_runs=selection.record["runs"],
                 selected_runs=[r for r in selection.record["runs"] if r["start_position"] in selected],
                 used_runs=runs, rejected_runs=rejected,
                 segments=segments, raw_segments=count, contributing_positions=sorted(used),
                 duration_seconds=len(used)/fs,
                 longest_run_seconds=max((r["used_duration_seconds"] for r in runs), default=0),
                 effective_averages=dict(value=count**2/(count+2*penalty) if count else None,
                     method="gaussian_white_noise_window_overlap_v1", approximate=True,
                     assumptions="equal-weight mean periodograms; Gaussian white noise",
                     reason=None if count else "no eligible segments", overlap_pair_square_sum=penalty))
    units = signal.metadata[variable]["unit"] + ("^2/Hz" if requested["scaling"] == "density" else "^2")
    config = dict(requested=requested, resolved=resolved)
    if count < requested["min_segments"] or len(used) < observations.min_points:
        return _result(signal, used_selection, observations, metric="spectral.welch_mean_v1", value=None,
            status="insufficient_support", reason="too few segments or contributing points for the explicit request",
            units=units, config=config, extra=extra)
    psd = weighted_psd / count
    if not np.isfinite(psd).all():
        return _result(signal, used_selection, observations, metric="spectral.welch_mean_v1", value=None,
            status="degenerate_input", reason="spectrum exceeds finite precision", units=units, config=config, extra=extra)
    extra.update(constant_input=bool(np.ptp(values[sorted(used)]) == 0),
                 zero_spectrum=bool(np.all(psd == 0)),
                 cycles_longest_run=frequency*extra["longest_run_seconds"])
    return _result(signal, used_selection, observations, metric="spectral.welch_mean_v1",
        value=dict(frequency_hz=frequency, psd=psd, physical_frequency_hz=physical_grid,
                   physical_psd=weighted_physical_psd/count), units=units, config=config, extra=extra)

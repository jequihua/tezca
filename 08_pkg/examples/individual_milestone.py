"""Cumulative installed M002 path; synthetic inputs, stdout only, no optional extras."""
import json

import numpy as np
import pandas as pd
from scipy.special import ndtri

from tezca.data import AnalysisSeries
from tezca.density_fisher import HistogramSpec, KDESpec, density_fisher, fit_density_reference
from tezca.fi1 import FI1Spec, fi1_discrete, fi1_windows, fit_fi1_reference
from tezca.harmonic import HarmonicSpec, calibrate_siegel, compatibility_fields, harmonic_diagnostics
from tezca.individual import (DailySpec, apply_daily_reference, autocorrelation,
                              describe_series, estimate_trend, fit_daily_reference)
from tezca.spectral import WelchSpec, power_spectrum
from tezca.spectral_summary import BandSpec, ModelSpec, PeakSpec, band_power, spectral_models, spectral_peaks


def make(values, cadence="1h"):
    values = np.asarray(values, dtype=float)
    return AnalysisSeries(values, time=pd.date_range("2000", periods=len(values), freq=cadence, tz="UTC"),
        variables=["x"], metadata={"x": {"unit": "m", "representation": "raw"}},
        source={"id": "individual-milestone-v1", "segment": "synthetic"},
        clock={"mode": "regular", "cadence": cadence}, observed=np.isfinite(values)[:, None])


def main():
    trend_data = make([1, 3, 5, 7], "1d")
    empty = make([np.nan]*4)
    training_data = make((np.arange(24)+np.array([0, 2, 100])[:, None]).ravel())
    original = training_data.values
    daily_ref = fit_daily_reference(training_data, spec=DailySpec(civil_timezone="UTC"), training_positions=list(range(48)))
    residual = apply_daily_reference(training_data, reference=daily_ref)
    residual_stats = describe_series(residual)
    residual_spectrum = power_spectrum(residual, variable="x", spec=WelchSpec(48))
    short_cycle = apply_daily_reference(make(np.arange(24)), spec=DailySpec(civil_timezone="UTC"))
    sine = make(np.sin(2*np.pi*np.arange(64)/8), "1s")
    spectrum = power_spectrum(sine, variable="x", spec=WelchSpec(64, taper="boxcar", overlap=0, detrend=False))
    peaks = spectral_peaks(spectrum, spec=PeakSpec())
    daily = make(np.sin(2*np.pi*np.arange(672)/24))
    daily_spectrum = power_spectrum(daily, variable="x", spec=WelchSpec(672))
    models = spectral_models(daily_spectrum, spec=ModelSpec())
    band = band_power(daily_spectrum, spec=BandSpec([20, 28]))
    short_spectrum = power_spectrum(make([1, 2]), variable="x", spec=WelchSpec(64))
    mass_data = make([0, .5, 1, 3])
    fi_ref = fit_fi1_reference(mass_data, variable="x", spec=FI1Spec([0, 1, 3]), training_positions=[0, 1, 2, 3])
    fi = fi1_discrete(mass_data, variable="x", reference=fi_ref)
    window_data = make(np.arange(168.) % 3)
    trajectory = fi1_windows(window_data, variable="x", anchor=window_data.time.index[0], spec=FI1Spec([0, 1, 3]))
    histogram = density_fisher(mass_data, variable="x", spec=HistogramSpec([0, 1, 3]))
    gaussian = make(2*ndtri((np.arange(4096)+.5)/4096))
    density_ref = fit_density_reference(gaussian, variable="x", spec=KDESpec(bandwidth=.3, bounds=[-16, 16]),
                                       training_positions=np.arange(4096))
    kde = density_fisher(gaussian, variable="x", reference=density_ref)
    harmonic_data = make(np.sin(2*np.pi*7*np.arange(129)/129))
    calibration = calibrate_siegel(m=64, replications=19999, seed=2026101101)
    harmonic = harmonic_diagnostics(harmonic_data, variable="x", spec=HarmonicSpec(), calibration=calibration)
    harmonic_missing = harmonic_diagnostics(harmonic_data, variable="x", spec=HarmonicSpec(backend="sensor_pinned"))
    constant = make([1, 1])
    unavailable = dict(
        descriptions=describe_series(empty)[0], trend=estimate_trend(make([1]), variable="x"),
        acf=autocorrelation(make([1]), variable="x"), daily=describe_series(short_cycle)[0],
        welch=short_spectrum, models=spectral_models(short_spectrum, spec=ModelSpec()),
        peaks=spectral_peaks(short_spectrum, spec=PeakSpec()), band=band_power(short_spectrum, spec=BandSpec([20, 28])),
        fi1=fi1_discrete(constant, variable="x", spec=FI1Spec([0, 1, 3])),
        histogram=density_fisher(constant, variable="x", spec=HistogramSpec([0, 1, 3])),
        kde=density_fisher(constant, variable="x", spec=KDESpec(bandwidth=.3, bounds=[-16, 16])),
        fisher=harmonic_missing["fisher"], siegel=harmonic_missing["siegel"])
    # Exercise ownership, then expose the checks to the installed-path assertions.
    residual_copy = residual.values
    residual_copy.iloc[0, 0] = -999
    daily_ref.describe()["variables"]["x"]["template"][0] = -999
    spectrum.value["psd"][:] = -999
    calibration.describe()["sorted_T"][:] = -999
    pd.testing.assert_frame_equal(training_data.values, original)
    assert residual.values.iloc[0, 0] == -1
    assert daily_ref.describe()["variables"]["x"]["template"][0] == 1
    assert (spectrum.value["psd"] >= 0).all()
    assert (calibration.describe()["sorted_T"] >= 0).all()
    refs = dict(daily=daily_ref.identity, residual=residual.reference_id,
        residual_description=residual_stats[0].summary()["reference_id"],
        residual_spectrum=residual_spectrum.summary()["reference_id"], fi1=fi_ref.identity,
        fi1_result=fi.summary()["reference_id"], density=density_ref.identity,
        density_result=kde.summary()["reference_id"], calibration=calibration.identity,
        siegel=harmonic["siegel"].summary()["reference_id"])
    support_ids = dict(spectrum=spectrum.summary()["support"]["support_id"],
        peaks=peaks.summary()["support"]["support_id"], daily=daily_spectrum.summary()["support"]["support_id"],
        models=models.summary()["support"]["support_id"], band=band.summary()["support"]["support_id"],
        fisher=harmonic["fisher"].summary()["support"]["support_id"],
        siegel=harmonic["siegel"].summary()["support"]["support_id"])
    output = dict(mean=describe_series(trend_data)[0].value, slope_per_day=estimate_trend(trend_data, variable="x").value,
        lag_seconds=autocorrelation(trend_data, variable="x").summary()["support"]["lag"]["realized_seconds"],
        template_first=float(daily_ref.describe()["variables"]["x"]["template"][0]),
        heldout_residual=residual.values.iloc[48:, 0].tolist(), residual_basis=residual_stats[0].summary()["representation"],
        peak_hz=float(peaks.value["table"].iloc[0]["frequency_hz"]), peak_psd=float(spectrum.value["psd"].max()),
        band_power=band.value["power"], model_status=models.summary()["compute_status"],
        model_grid=models.summary()["support"]["fit_grid"], fi1=fi.value["fi1"], entropy=fi.value["entropy_nats"],
        windows=len(trajectory.table), histogram_J=histogram.value["J_raw"], kde_J=kde.value["J_raw"], kde_mass=kde.value["integrated_mass"],
        harmonic=compatibility_fields(harmonic)["fields"], harmonic_p={k: v.value["p_value"] for k, v in harmonic.items()},
        harmonic_missing=compatibility_fields(harmonic_missing)["fields"],
        unavailable={k: dict(status=v.summary()["compute_status"], value=v.value) for k, v in unavailable.items()},
        references=refs, support_ids=support_ids, input_unchanged=True, owned_outputs=True)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()

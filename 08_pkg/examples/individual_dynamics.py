"""Run against an installed wheel; no files or optional features required."""
import json

import numpy as np
import pandas as pd
from scipy.special import ndtri

from tezca.data import AnalysisSeries
from tezca.individual import (DailySpec, apply_daily_reference, autocorrelation,
                              describe_series, estimate_trend, fit_daily_reference)
from tezca.spectral import WelchSpec, power_spectrum
from tezca.spectral_summary import (ModelSpec, PeakSpec, BandSpec, spectral_models,
                                    spectral_peaks, band_power)
from tezca.fi1 import FI1Spec, fi1_discrete, fi1_windows
from tezca.density_fisher import HistogramSpec, KDESpec, density_fisher


def make(values, cadence="1h"):
    return AnalysisSeries(values, time=pd.date_range("2000", periods=len(values), freq=cadence, tz="UTC"),
        variables=["x"], metadata={"x": {"unit": "m", "representation": "raw"}},
        source={"id": "individual-example-v1", "segment": "synthetic"},
        clock={"mode": "regular", "cadence": cadence}, observed=np.ones((len(values), 1), dtype=bool))


def main():
    data = make([1, 3, 5, 7], "1d")
    slope = estimate_trend(data, variable="x").value
    stats = describe_series(data)
    repeated = make((np.arange(24) + np.array([0, 2, 100])[:, None]).ravel())
    original = repeated.values
    ref = fit_daily_reference(repeated, spec=DailySpec(civil_timezone="UTC"), training_positions=list(range(48)))
    residual = apply_daily_reference(repeated, reference=ref)
    pd.testing.assert_frame_equal(original, repeated.values)
    short = apply_daily_reference(make(np.arange(24)), spec=DailySpec(civil_timezone="UTC"))
    assert short.values.isna().all().all()
    sine = power_spectrum(make(np.sin(2*np.pi*np.arange(64)/8), "1s"), variable="x",
                          spec=WelchSpec(64, taper="boxcar", overlap=0, detrend=False))
    daily_spectrum = power_spectrum(make(np.sin(2*np.pi*np.arange(672)/24)), variable="x", spec=WelchSpec(672))
    daily_band = band_power(daily_spectrum, spec=BandSpec([20, 28]))
    models = spectral_models(daily_spectrum, spec=ModelSpec())
    partition = FI1Spec([0, 1, 3])
    mass = fi1_discrete(make([0, .5, 1, 3]), variable="x", spec=partition)
    window_data = make(np.arange(168.) % 3)
    trajectory = fi1_windows(window_data, variable="x", anchor=window_data.time.index[0], spec=partition)
    unavailable_fi1 = fi1_discrete(make([1, 1]), variable="x", spec=partition)
    density_histogram = density_fisher(make([0, .5, 1, 3]), variable="x", spec=HistogramSpec([0, 1, 3]))
    density_kde = density_fisher(make(2*ndtri((np.arange(4096)+.5)/4096)), variable="x",
                                spec=KDESpec(bandwidth=.3, bounds=[-16, 16]))
    unavailable_density = density_fisher(make([1, 1]), variable="x", spec=HistogramSpec([0, 1, 3]))
    output = dict(slope_per_day=slope, mean=stats[0].value,
        lag_seconds=autocorrelation(data, variable="x").summary()["support"]["lag"]["realized_seconds"],
        template_first=float(ref.describe()["variables"]["x"]["template"][0]),
        heldout_residual=residual.values.iloc[48:, 0].tolist(),
        insufficient_cycle=describe_series(short)[0].summary()["compute_status"],
        raw_basis=stats[0].summary()["representation"], residual_basis=describe_series(residual)[0].summary()["representation"],
        reference_id=ref.identity, input_unchanged=True,
        fi1=dict(value=mass.value["fi1"], entropy_nats=mass.value["entropy_nats"],
                 windows=len(trajectory.table), eligible_windows=trajectory.describe()["summary"]["eligible_windows"],
                 unavailable=unavailable_fi1.summary()["compute_status"]),
        density=dict(histogram_J=density_histogram.value["J_raw"], kde_J=density_kde.value["J_raw"],
                     kde_mass=density_kde.value["integrated_mass"],
                     unavailable=unavailable_density.summary()["compute_status"]),
        spectral=dict(bins=len(sine.value["frequency_hz"]),
                      peak_hz=float(sine.value["frequency_hz"][np.argmax(sine.value["psd"])]),
                      peak_psd=float(sine.value["psd"].max()),
                      integrated_power=float(sine.value["psd"].sum()/64),
                      ranked_peak_hz=float(spectral_peaks(sine, spec=PeakSpec()).value["table"].iloc[0]["frequency_hz"]),
                      model_grid=models.summary()["support"]["fit_grid"],
                      daily_band_status=daily_band.summary()["compute_status"], daily_band_power=daily_band.value["power"]))
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()

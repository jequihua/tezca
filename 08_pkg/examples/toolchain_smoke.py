"""Installed-toolchain interoperability example, not a Tezca scientific API."""

import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.signal import welch

import tezca


def main():
    values = np.sin(2 * np.pi * 0.125 * np.arange(64, dtype=float))
    original = values.copy()
    frame = pd.DataFrame(
        {"signal": values},
        index=pd.date_range("2000-01-01", periods=64, freq="s", tz="UTC"),
    )
    frequencies, power = welch(
        frame["signal"].to_numpy(), fs=1.0, window="boxcar", nperseg=64,
        noverlap=0, nfft=64, detrend=False, return_onesided=True,
        scaling="density", axis=-1, average="mean",
    )
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(frame["signal"].to_numpy(), original)
    print(json.dumps({
        "versions": {"python": platform.python_version(), "tezca": tezca.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__},
        "samples": len(frame), "timezone": str(frame.index.tz),
        "elapsed_seconds": (frame.index[-1] - frame.index[0]).total_seconds(),
        "frequency_bins": len(frequencies),
        "peak_hz": float(frequencies[np.argmax(power)]),
        "integrated_power": float(np.sum(power) * (frequencies[1] - frequencies[0])),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

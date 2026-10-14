"""Portable installed-package controls using synthetic inputs and stdlib unittest."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from tezca.data import AnalysisSeries
from tezca.individual import describe_series
from tezca.spectral import WelchSpec, power_spectrum
from tezca.spectral_summary import BandSpec, band_power
from tezca.exports import ResultItem, export_results, load_results
from tezca.support import ObservationPolicy


def series(values, *, observed=True):
    return AnalysisSeries(values, variables=['x'],
        time=pd.date_range('2000-01-01', periods=len(values), freq='h', tz='UTC'),
        metadata={'x':dict(unit='m', representation='raw')},
        source=dict(id='portable-contract', segment='synthetic'),
        observed=np.ones(len(values),dtype=bool) if observed else None,
        clock=dict(mode='regular',cadence='1h'))


class InstalledContractTests(unittest.TestCase):
    def test_owned_values_and_literal_mean(self):
        values=np.array([1.,2.,3.,4.])
        data=series(values)
        values[0]=999
        view=data.values; view.iloc[1,0]=999
        np.testing.assert_array_equal(data.values.to_numpy().ravel(),[1.,2.,3.,4.])
        mean=next(r for r in describe_series(data) if r.summary()['method_id']=='individual.mean_v1')
        self.assertEqual(mean.value,2.5)
        self.assertEqual(mean.summary()['compute_status'],'ok')
        self.assertEqual(mean.summary()['support']['positions'],[0,1,2,3])

    def test_unknown_provenance_is_not_an_observation(self):
        data=series([1.,2.,3.,4.], observed=False)
        for mode,compute,use in [('observed_only','insufficient_support','insufficient_support'),
                                 ('provided','ok','unknown_provenance')]:
            with self.subTest(mode=mode):
                mean=next(r for r in describe_series(data,observations=ObservationPolicy(mode))
                          if r.summary()['method_id']=='individual.mean_v1')
                self.assertEqual(mean.summary()['compute_status'],compute)
                self.assertEqual(mean.summary()['use_status'],use)

    def test_literal_spectrum_and_withheld_export_reload(self):
        data=series(np.cos(2*np.pi*np.arange(64)/8))
        spectrum=power_spectrum(data,variable='x',spec=WelchSpec(32,overlap=16,taper='boxcar',detrend=False))
        frequencies=spectrum.value['frequency_hz']; power=spectrum.value['psd']
        self.assertAlmostEqual(float(frequencies[4]),1/(8*3600),delta=1e-16)
        self.assertAlmostEqual(float(power[4]),57600.,delta=1e-8)
        self.assertAlmostEqual(float(power.sum())/(32*3600),.5,delta=1e-12)
        withheld=band_power(spectrum,spec=BandSpec([48,240]))
        self.assertIsNone(withheld.value)
        self.assertEqual(withheld.summary()['compute_status'],'outside_frequency_support')
        self.assertEqual(withheld.summary()['reason'],'band/denominator would require extrapolation')
        with tempfile.TemporaryDirectory(prefix='tezca-public-test-') as name:
            destination=Path(name)/'result'
            manifest=export_results(destination,sources={'signal':data},results={
                'spectrum':ResultItem('spectrum',spectrum,sources=['signal']),
                'withheld':ResultItem('metric',withheld,sources=['signal'])},
                requested_policy={'example':'portable'},resolved_policy={'example':'portable'})
            loaded=load_results(destination,expected_bundle_id=manifest['bundle_id'],
                expected_sources=manifest['source_identities'],expected_references=manifest['reference_identities'])
            self.assertEqual(loaded.manifest['bundle_id'],manifest['bundle_id'])
            saved=loaded.details['results']['withheld']['record']
            self.assertIsNone(saved['value'])
            self.assertEqual(saved['compute_status'],'outside_frequency_support')
            with self.assertRaises(ValueError):
                load_results(destination,expected_bundle_id='0'*64)


if __name__=='__main__': unittest.main()

"""Installed models/sensitivity path with the original summary and an S01 bundle.

--write summary.json also creates summary-results/; --validate retains the
independent stdlib check; --reload-results validates the bundle without models.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(summary):
    return hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate(path):
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    if set(record)!={'schema','summary','sha256'} or record['schema']!='tezca.m004.example/1':
        raise ValueError('invalid example summary schema')
    summary=record['summary']
    if digest(summary)!=record['sha256']:
        raise ValueError('digest mismatch')
    assert summary['literal']==dict(density=32.0,peak_hz=.125,power=.5,segments=8)
    assert summary['kmeans']['centers']==[-2.0,2.0] and abs(summary['kmeans']['inertia']-4)<1e-10
    assert summary['dbscan']['clusters']==2 and summary['dbscan']['noise']==1
    assert summary['statuses']==dict(insufficient='insufficient_support',unknown='unknown_provenance',incomparable='incomparable')
    assert summary['assignment']['reference_id']==summary['assignment']['training_reference_id']
    assert summary['assignment']['row_ids']==['future0','future1']
    assert summary['sensitivity']['routes']==5 and summary['sensitivity']['contrasts']==6
    assert summary['timeline']['barrier_rows']==['timeline2']
    assert not any(name=='tezca' or name.startswith('tezca.') for name in sys.modules)
    print('Validated summary and exact support/reference/row links without Tezca or refitting.')


def result_evidence(items):
    """Small independent before/after witness of native labels and frozen fits."""
    records = {name: item.describe() for name, item in items.items()}
    return saved_evidence(records)


def saved_evidence(records):
    evidence = {}
    for name, item in records.items():
        if item['kind'] != 'regimes':
            continue
        reference = item['reference']
        rows = [{key: value.isoformat() if key in ('start', 'end') else
                 {'nonfinite': 'nan'} if isinstance(value, float) and math.isnan(value) else value
                 for key, value in row.items()} for row in item['rows']]
        timeline = item['timeline']
        evidence[name] = dict(identity=item['record']['identity'],
            reference_id=item['record']['reference_id'], rows=rows,
            training_ids=reference['support']['row_ids'],
            preprocess_state={key: values.tolist() for key, values in reference.get('preprocess_state', {}).items()},
            # The saved timeline is a set of numerical tables, never a fitted model.
            timeline=None if timeline is None else {
                key: json.loads(frame.to_json(orient='split',date_format='iso'))
                for key, frame in timeline.items() if key in ('rows','episodes','transitions')})
    return evidence


def reload_results(path):
    from tezca.exports import load_results
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if digest(record['summary']) != record['sha256']:
        raise ValueError('digest mismatch')
    expected = record['summary']['result_bundle']
    bundle = load_results(Path(path).with_name(Path(path).stem + '-results'),
        expected_bundle_id=expected['bundle_id'], expected_sources=expected['source_identities'],
        expected_references=expected['reference_identities'])
    assert saved_evidence(bundle.details['results']) == expected['regimes']
    assert not any(name == 'sklearn' or name.startswith('sklearn.') for name in sys.modules)
    print('Validated S01 result bundle, native labels, frozen fits and gap timelines without sklearn or refitting.')


def build(bundle_directory=None):
    import runpy
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from tezca.data import AnalysisSeries
    from tezca.exports import ResultItem, export_results
    from tezca.fi1 import FI1Spec, fit_fi1_reference, fi1_discrete
    from tezca.irregular import IrregularSpec, irregular_spectrum
    from tezca.regimes import RegimeFeatures, ClusterSpec, CapabilityError, cluster_regimes, assign_regimes, summarize_regime_timeline
    from tezca.sensitivity import spectral_sensitivity, compare_metric_results
    from tezca.support import ObservationPolicy

    bank=runpy.run_path(str(Path(__file__).with_name('reconstruction_sensitivity.py')))
    training,complete,cases=bank['fixed_bank']()
    _,masked,draws=cases[3]
    compared=spectral_sensitivity(complete,masked,draws,variables=['x'],**bank['settings']())
    assert set(compared.routes.compute_status)=={'ok'}
    assert compared.routes.query("route=='Wc'").source_id.tolist()==[draws['plus'].identity,draws['minus'].identity]
    assert compared.routes.query("route=='Im'").iloc[0].spectrum['support']['contributing_positions']
    start=pd.Timestamp('2000-03-01',tz='UTC')
    n=np.arange(512)
    signal=AnalysisSeries(np.sin(2*np.pi*8*n/64),variables=['x'],time=pd.date_range(start,periods=512,freq='s'),
        metadata={'x':dict(unit='m',representation='raw')},source=dict(id='literal-sine',segment='synthetic'),
        clock=dict(mode='regular',cadence='1s'),observed=np.ones(512,dtype=bool))
    spec=IrregularSpec(interval=[start,start+pd.Timedelta('512s')],segment='64s',hop='64s',anchor=start,
        frequency_hz=np.arange(1,32)/64,requested_band_hz=[1/64,31/64],f_ref_hz=1)
    spectrum=irregular_spectrum(signal,variable='x',spec=spec)
    psd=spectrum.value['psd']
    assert abs(psd[7]-32)<1e-10 and abs(psd.sum()/64-.5)<1e-10
    assert spectrum.summary()['support']['positions']==list(range(512))
    missing=bank['masked_copy'](complete,np.ones((672,3),dtype=bool))
    insufficient=irregular_spectrum(missing,variable='x',spec=bank['settings']()['irregular']).summary()['compute_status']

    sources = dict(signal=signal, complete=complete, masked=masked, plus=draws['plus'], minus=draws['minus'])
    def feature(values, *, prefix='train',unknown=False,times=None):
        values=np.asarray(values,dtype=float)
        times=pd.date_range(start,periods=len(values),freq='h') if times is None else times
        data=AnalysisSeries(values,variables=['coordinate'],time=times,
            metadata={'coordinate':dict(unit='1',representation='raw')},source=dict(id=prefix,segment='synthetic'),
            clock=dict(mode='irregular'),observed=None if unknown else np.isfinite(values))
        sources[prefix] = data
        return RegimeFeatures(data,features=['coordinate'],row_ids=[prefix+str(i) for i in range(len(values))],
            definitions={'coordinate':dict(method_id='synthetic_coordinate_v1',reference_id=None,window={'kind':'point'})})
    settings=ClusterSpec('kmeans',n_clusters=2,n_init=10,random_state=74003,algorithm='lloyd')
    points=feature([-3,-2,-1,1,2,3])
    literal=cluster_regimes(points,clusterer=settings,preprocess=None)
    centers=sorted(literal.reference.describe()['centroids'].ravel())
    inertia=literal.reference.describe()['diagnostics']['inertia']
    np.testing.assert_allclose(centers,[-2,2],rtol=0,atol=1e-10)
    assert abs(inertia-4)<1e-10
    frozen=cluster_regimes(points,clusterer=settings,preprocess=StandardScaler(),fit_scope='training_only',training_ids=list(points.values.index))
    future=feature([-4,4],prefix='future',times=pd.date_range(start+pd.Timedelta('24h'),periods=2,freq='h'))
    assigned=assign_regimes(future,reference=frozen.reference)
    assert assigned.rows.native_label.iloc[0]!=assigned.rows.native_label.iloc[1]
    assert frozen.reference.fitted_preprocess.mean_[0]==0
    unknown_result=cluster_regimes(feature([-3,-2,-1,1,2,3],prefix='unknown',unknown=True),clusterer=settings,preprocess=None,
        observations=ObservationPolicy('provided'))
    unknown=unknown_result.describe()['use_status']
    density=cluster_regimes(feature([0,.05,1,1.05,5],prefix='density'),clusterer=ClusterSpec('dbscan',eps=.1,min_samples=2,metric='euclidean'),preprocess=None)
    try:
        assign_regimes(future,reference=density.reference)
    except CapabilityError:
        refusal=True
    else:
        raise AssertionError('DBSCAN invented assignment')
    refs=[fit_fi1_reference(training,variable='x',spec=FI1Spec(edges),training_positions=range(672))
          for edges in ([-4,0,4],[-4,-1,0,1,4])]
    fi=[fi1_discrete(complete,variable='x',reference=r) for r in refs]
    interval=bank['settings']()['irregular'].record['interval']
    mismatch=compare_metric_results(*fi,baseline_interval=interval,candidate_interval=interval)
    incomparable=mismatch.iloc[0].compute_status
    timed=feature([-4,-4,np.nan,-4,4,4],prefix='timeline',
        times=pd.DatetimeIndex([start+pd.Timedelta(hours=h) for h in (0,1,2,10,11,12)]))
    timed_result=assign_regimes(timed,reference=frozen.reference)
    timeline=summarize_regime_timeline(timed_result,max_gap='1h',noise_policy='barrier')
    assert len(timeline['episodes'])==3 and len(timeline['transitions'])==1
    summary=dict(literal=dict(density=round(float(psd[7]),10),peak_hz=.125,power=round(float(psd.sum()/64),10),segments=8),
        sensitivity=dict(report_id=compared.identity,routes=len(compared.routes),contrasts=len(compared.contrasts),
            source_ids=compared.routes.source_id.tolist(),draw_ids=compared.routes.draw_id.tolist()),
        kmeans=dict(centers=[round(float(c),10) for c in centers],inertia=inertia),
        assignment=dict(reference_id=assigned.reference.identity,training_reference_id=frozen.reference.identity,
            row_ids=assigned.rows.row_id.tolist(),training_ids=frozen.reference.describe()['support']['row_ids']),
        dbscan=dict(clusters=density.describe()['cluster_count'],noise=density.describe()['noise'],assignment_refused=refusal),
        timeline=dict(episodes=len(timeline['episodes']),transitions=len(timeline['transitions']),
            barrier_rows=timeline['rows'].query("status=='excluded_missing'").row_id.tolist()),
        statuses=dict(insufficient=insufficient,unknown=unknown,incomparable=incomparable),
        qualification='Synthetic method qualification only; no ecological truth, natural-gap RMSE or confidence claim.')
    if bundle_directory is not None:
        results = {
            'spectrum': ResultItem('spectrum', spectrum, sources=['signal']),
            'sensitivity': ResultItem('sensitivity', compared, sources=['complete','masked','plus','minus']),
            'literal': ResultItem('regimes', literal, sources=['train']),
            'frozen': ResultItem('regimes', frozen, sources=['train']),
            'future': ResultItem('regimes', assigned, sources=['future']),
            'unknown': ResultItem('regimes', unknown_result, sources=['unknown']),
            'dbscan': ResultItem('regimes', density, sources=['density']),
            'timeline': ResultItem('regimes', timed_result, sources=['timeline'], timeline=timeline),
        }
        manifest = export_results(bundle_directory, results=results, sources=sources,
            references=[literal.reference, frozen.reference, unknown_result.reference, density.reference],
            requested_policy=dict(example='installed_models_and_sensitivity'))
        summary['result_bundle'] = dict(bundle_id=manifest['bundle_id'],
            source_identities=manifest['source_identities'], reference_identities=manifest['reference_identities'],
            regimes=result_evidence(results))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write',type=Path)
    group.add_argument('--validate',type=Path)
    group.add_argument('--reload-results',type=Path)
    args=parser.parse_args()
    if args.validate:
        validate(args.validate)
    elif args.reload_results:
        reload_results(args.reload_results)
    else:
        summary=build(args.write.with_name(args.write.stem + '-results'))
        with args.write.open('x',encoding='utf-8') as output:
            json.dump(dict(schema='tezca.m004.example/1',summary=summary,sha256=digest(summary)),output,sort_keys=True,allow_nan=False)
        print('Saved synthetic M004 summary; validate it in a new process.')

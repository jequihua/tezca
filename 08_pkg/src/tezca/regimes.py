"""Explicit candidate clusters, frozen assignment and unsmoothed support timelines."""

from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ._owned import identity, owned, text_field
from .data import AnalysisSeries
from .individual import _ColumnSignal
from .support import ObservationPolicy, _aware_timestamp, positive_duration, select_support


class CapabilityError(ValueError):
    pass


class RegimeFitError(RuntimeError):
    """Custom backends may raise this for an expected numerical fit failure."""


class RegimeFeatures:
    """Owned AnalysisSeries plus explicit feature definitions and original row IDs."""
    def __init__(self, data, *, features, definitions, row_ids, intervals=None):
        if not isinstance(data, AnalysisSeries):
            raise ValueError('features require AnalysisSeries')
        features, row_ids = list(features), list(row_ids)
        if not features or len(set(features)) != len(features) or any(f not in data.metadata for f in features):
            raise ValueError('explicit distinct known features required')
        if [f for f in data.values if f in features] != features:
            raise ValueError('feature order must match source order')
        if len(row_ids) != len(data.values) or len(set(row_ids)) != len(row_ids):
            raise ValueError('row IDs must uniquely align with data')
        for row in row_ids:
            text_field(row,'row ID')
        if not isinstance(definitions,dict) or set(definitions) != set(features):
            raise ValueError('each selected feature needs a definition')
        schema = []
        for feature in features:
            definition = definitions[feature]
            if not isinstance(definition,dict) or set(definition) != {'method_id','reference_id','window'}:
                raise ValueError('feature definitions require method_id, reference_id and window')
            text_field(definition['method_id'],'metric definition')
            if definition['reference_id'] is not None:
                text_field(definition['reference_id'],'feature reference')
            if not isinstance(definition['window'],dict) or not definition['window']:
                raise ValueError('explicit feature window definition required')
            schema.append(dict(name=feature,unit=data.metadata[feature]['unit'],
                basis=data.metadata[feature]['representation'],**owned(definition)))
        starts = list(data.time.index)
        ends = starts if intervals is None else list(intervals)
        if len(ends) != len(starts):
            raise ValueError('interval ends must align with rows')
        ends = [_aware_timestamp(t,'interval end') for t in ends]
        if any(e < s or (intervals is not None and e == s) for s,e in zip(starts,ends)):
            raise ValueError('interval ends must follow starts')
        self._data, self._features = data, features
        self._schema, self._ids = owned(schema), row_ids.copy()
        self._geometry = [dict(row_id=r,start=s,end=e) for r,s,e in zip(row_ids,starts,ends)]
        self._identity = identity(dict(input=data.identity,schema=schema,rows=self._geometry))

    @property
    def identity(self):
        return self._identity

    @property
    def schema(self):
        return owned(self._schema)

    @property
    def values(self):
        frame = self._data.values[self._features]
        frame.index = self._ids
        return frame

    def describe(self):
        return dict(identity=self.identity,source_id=self._data.identity,source=self._data.source,
                    schema=self.schema,rows=owned(self._geometry))


class ClusterSpec:
    def __init__(self, method, **parameters):
        if method == 'kmeans':
            required = {'n_clusters','n_init','random_state'}
            allowed = required | {'init','max_iter','tol','algorithm'}
            defaults = dict(init='k-means++',max_iter=300,tol=1e-4,algorithm='lloyd')
        elif method == 'dbscan':
            required = {'eps','min_samples','metric'}
            allowed = required | {'metric_params','algorithm','leaf_size','p','n_jobs'}
            defaults = dict(metric_params=None,algorithm='auto',leaf_size=30,p=None,n_jobs=None)
        else:
            raise ValueError('method must be kmeans or dbscan; use CustomClusterer for custom methods')
        if not required <= set(parameters) or set(parameters)-allowed:
            raise ValueError(f'{method} requires {sorted(required)} and rejects unrelated parameters')
        settings = dict(defaults,**owned(parameters))
        if method == 'kmeans':
            if any(type(settings[k]) is not int or settings[k]<1 for k in ('n_clusters','n_init','max_iter')):
                raise ValueError('K, restarts and iterations must be positive integers')
            if type(settings['random_state']) is not int or not 0 <= settings['random_state'] < 2**32:
                raise ValueError('explicit integer random_state required')
            if settings['init'] not in {'k-means++','random'} or settings['algorithm'] not in {'lloyd','elkan'}:
                raise ValueError('unsupported KMeans initialization or algorithm')
            if isinstance(settings['tol'],bool) or not np.isfinite(settings['tol']) or settings['tol']<0:
                raise ValueError('tol must be finite nonnegative')
        else:
            if isinstance(settings['eps'],bool) or not np.isfinite(settings['eps']) or settings['eps']<=0:
                raise ValueError('eps must be finite positive')
            if type(settings['min_samples']) is not int or settings['min_samples']<1:
                raise ValueError('min_samples must be a positive integer')
            text_field(settings['metric'],'metric')
        self._record = dict(method=method,parameters=settings)

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


class CustomClusterer:
    def __init__(self, factory, *, config, version, assignment=False, sample_weight=False,
                 soft=False, precomputed=False, randomness=False, noise_labels=(), soft_meaning=None):
        if not callable(factory) or not isinstance(config,dict):
            raise ValueError('custom factory and configuration mapping required')
        text_field(version,'backend version')
        capabilities = dict(assignment=assignment,sample_weight=sample_weight,soft=soft,
                            precomputed=precomputed,randomness=randomness)
        if any(type(v) is not bool for v in capabilities.values()):
            raise ValueError('capabilities must be booleans')
        if soft:
            text_field(soft_meaning,'method-specific membership meaning')
        elif soft_meaning is not None:
            raise ValueError('membership meaning requires soft capability')
        if not randomness and 'random_state' in config:
            raise CapabilityError('custom randomness capability is not declared')
        self._factory = factory
        self._record = dict(method='custom',parameters=owned(config),version=version,
            capabilities=capabilities,noise_labels=list(noise_labels),soft_meaning=soft_meaning)
        identity(self._record)  # Require explicit serializable scientific configuration.


class CustomPreprocess:
    def __init__(self, factory, *, config, version, row_map=False):
        if not callable(factory) or not isinstance(config,dict) or type(row_map) is not bool:
            raise ValueError('custom preprocessing requires factory/config and explicit row_map')
        text_field(version,'transformer version')
        self._factory = factory
        self._record = dict(method='custom',parameters=owned(config),version=version,row_map=row_map)
        identity(self._record)


def _factory_instance(adapter):
    first, second = adapter._factory(), adapter._factory()
    if first is second or any(hasattr(first,k) for k in ('labels_','cluster_centers_','n_features_in_')):
        raise ValueError('custom factories must return independent unfitted instances')
    return first


def _sklearn():
    try:
        import sklearn
    except ImportError as error:
        raise ImportError('This request needs the optional models extra: install tezca[models]') from error
    return sklearn


def _clusterer(clusterer):
    if isinstance(clusterer,CustomClusterer):
        model = _factory_instance(clusterer)
        record = dict(owned(clusterer._record),backend=type(model).__module__+'.'+type(model).__qualname__)
        if record['capabilities']['assignment'] and not callable(getattr(model,'predict',None)):
            raise CapabilityError('declared assignment requires genuine predict')
        return model,record
    sklearn = _sklearn()
    from sklearn.base import clone
    from sklearn.cluster import KMeans, DBSCAN
    if isinstance(clusterer,ClusterSpec):
        record = clusterer.record
        model = (KMeans if record['method']=='kmeans' else DBSCAN)(**record['parameters'])
    elif isinstance(clusterer,(KMeans,DBSCAN)):
        model = clone(clusterer)
        record = dict(method='kmeans' if isinstance(model,KMeans) else 'dbscan',parameters=model.get_params(deep=False))
    else:
        raise ValueError('use a supported sklearn estimator or explicit CustomClusterer')
    is_kmeans = record['method']=='kmeans'
    record.update(version=sklearn.__version__,backend='sklearn',noise_labels=[] if is_kmeans else [-1],soft_meaning=None,
        capabilities=dict(assignment=is_kmeans,sample_weight=True,soft=False,
            precomputed=not is_kmeans and record['parameters']['metric']=='precomputed',randomness=is_kmeans))
    identity(record)
    return model,record


def _preprocessor(preprocess):
    if preprocess is None:
        return None,dict(method='none',row_map=False)
    if isinstance(preprocess,CustomPreprocess):
        return _factory_instance(preprocess),owned(preprocess._record)
    sklearn = _sklearn()
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler
    if not isinstance(preprocess,StandardScaler):
        raise CapabilityError('native preprocessing supports StandardScaler; use CustomPreprocess with row identities otherwise')
    transformer = clone(preprocess)
    if not callable(getattr(transformer,'fit',None)) or not callable(getattr(transformer,'transform',None)):
        raise ValueError('preprocessing requires fit and transform')
    record = dict(method=type(transformer).__module__+'.'+type(transformer).__name__,
        parameters=transformer.get_params(deep=False),version=sklearn.__version__,row_map=False)
    identity(record)
    return transformer,record


def _transform(transformer, record, frame, *, fit):
    if transformer is None:
        return frame.copy()
    if fit:
        transformer.fit(frame.copy())
    result = transformer.transform(frame.copy())
    if record['method']=='custom':
        if not isinstance(result,pd.DataFrame):
            raise ValueError('custom transforms must return a DataFrame with explicit row identity map')
        if not result.index.is_unique or not set(result.index)<=set(frame.index):
            raise ValueError('transform row map must be a unique subset of input IDs')
        if not record['row_map'] and not result.index.equals(frame.index):
            raise ValueError('unannounced transform row count/order change')
        result = result.copy()
    else:
        if np.asarray(result).ndim!=2 or len(result)!=len(frame):
            raise ValueError('transform must preserve rows')
        result = pd.DataFrame(result,index=frame.index)
    if not len(result.columns) or not result.columns.is_unique or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError('transform must return finite numeric features')
    return result


def _selection(features, observations, requested=None):
    if not isinstance(features,RegimeFeatures) or not isinstance(observations,ObservationPolicy):
        raise ValueError('RegimeFeatures and ObservationPolicy required')
    data = features._data
    columns = [list(data.values).index(f) for f in features._features]
    finite = np.isfinite(features.values.to_numpy(dtype=float)).all(axis=1)
    qualified = data.eligible(observations)[:,columns].all(axis=1)
    flags = data.qualification
    qualified &= ~(flags['qc_rejected'][:,columns] | flags['structural'][:,columns]).any(axis=1)
    statuses = np.where(~finite,'excluded_missing',np.where(qualified,'assigned','excluded_quality'))
    if requested is not None:
        statuses[[r not in requested for r in features._ids]] = 'excluded_quality'
    return statuses.astype(object)


def _support(features, ids, observations):
    positions = [i for i,r in enumerate(features._ids) if r in set(ids)]
    records, uses = {},[]
    for feature in features._features:
        signal = _ColumnSignal(features._data,feature)
        selected = select_support(signal,observations=observations,policy='complete_case',estimator='point',within=positions)
        records[feature] = signal.support_record(selected)
        uses.append(signal.use(positions,observations,'exploratory_candidate_regimes'))
    use = 'unknown_provenance' if 'unknown_provenance' in uses else 'sensitivity_only' if positions else 'insufficient_support'
    return dict(row_ids=list(ids),positions=positions,variables=records,observations=observations.__dict__,use_status=use)


class RegimeReference:
    def __init__(self, record, model, preprocess):
        self._record = owned(record)
        self._identity = identity(record)
        self._model, self._preprocess = owned(model),owned(preprocess)

    @property
    def identity(self):
        return self._identity

    def describe(self):
        return owned(self._record)

    @property
    def fitted_model(self):
        return owned(self._model)

    @property
    def fitted_preprocess(self):
        return owned(self._preprocess)


def _records_frame(records, columns=None):
    records = owned(records)
    frame = pd.DataFrame(records,columns=columns)
    for column in ('native_label','from_label','to_label'):
        if column in frame:
            # Preserve mixed strings/numbers, exact large integers and missing None.
            frame[column] = pd.Series([r[column] for r in records],dtype=object)
    return frame


class RegimeResult:
    def __init__(self, rows, description, reference, coordinates, distances=None, cost_seconds=0):
        self._rows, self._description = owned(rows),owned(description)
        self._reference = reference
        self._coordinates, self._distances = owned(coordinates),owned(distances)
        self._cost = float(cost_seconds)
        self._identity = identity(dict(rows=rows,description=description,reference=reference.identity))

    @property
    def identity(self):
        return self._identity

    @property
    def rows(self):
        return _records_frame(self._rows,['row_id','start','end','native_label','display_label','status','reason'])

    @property
    def reference(self):
        return self._reference

    def describe(self):
        return dict(owned(self._description),identity=self.identity,cost_seconds=self._cost)


def _result(features, statuses, labels, ids, reference, coordinates, observations, *, status, reason,
            membership=None, distances=None, cost_seconds=0):
    labels = np.asarray(labels,dtype=object)
    if labels.ndim!=1 or len(labels)!=len(ids):
        raise ValueError('native labels must align with transformed row IDs')
    native = [label.item() if isinstance(label,np.generic) else label for label in labels.tolist()]
    if any(not isinstance(v,(str,int,float)) or isinstance(v,bool) or isinstance(v,float) and not np.isfinite(v) for v in native):
        raise ValueError('native labels must be finite scalar strings or numbers')
    frozen = reference.describe()
    noise_labels = frozen['clusterer']['noise_labels']
    mapping = {}
    for label in native:
        if label not in noise_labels and label not in mapping:
            mapping[label] = len(mapping)
    by_id = dict(zip(ids,native))
    rows = []
    for p,geometry in enumerate(features._geometry):
        label = by_id.get(geometry['row_id'])
        row_status = statuses[p]
        if geometry['row_id'] in by_id:
            row_status = 'noise' if label in noise_labels else 'assigned'
        elif row_status=='assigned':
            row_status = 'failed' if status!='ok' else 'excluded_quality'
        row_reason = None
        if row_status=='excluded_missing':
            row_reason = 'incomplete selected feature vector'
        elif row_status=='excluded_quality':
            row_reason = ('outside declared training support' if geometry['row_id'] not in frozen['requested_training_ids']
                and features.identity==frozen['source']['identity'] else
                'transformer row map excluded row' if statuses[p]=='assigned' else 'observation policy or quality exclusion')
        elif row_status=='failed':
            row_reason = reason
        rows.append(dict(geometry,native_label=label,display_label=mapping.get(label),status=row_status,reason=row_reason))
    counts = {label:sum(row['status']=='assigned' and row['native_label']==label for row in rows) for label in mapping}
    n = len(rows)
    support = _support(features,ids,observations)
    use = 'unknown_provenance' if 'unknown_provenance' in (support['use_status'],frozen['use_status']) else support['use_status']
    description = dict(method_id='regimes.candidate_clusters_v1',compute_status=status,reason=reason,
        metric_status='exploratory_candidate_regimes',use_status=use,source=features.describe(),schema=features.schema,
        reference_id=reference.identity,training=frozen,support=support,cluster_count=len(counts),cluster_sizes=counts,
        denominator=n,assigned=sum(r['status']=='assigned' for r in rows),noise=sum(r['status']=='noise' for r in rows),
        excluded=sum(r['status'].startswith('excluded') for r in rows),failed=sum(r['status']=='failed' for r in rows),
        display_mapping=mapping,membership=membership,soft_meaning=frozen['clusterer']['soft_meaning'],
        warning='Candidate clusters are descriptive; seasonality and overlap do not prove ecological regimes or change points.')
    description['fractions'] = {key:description[key]/n if n else None for key in ('assigned','noise','excluded','failed')}
    return RegimeResult(rows,description,reference,coordinates,distances,cost_seconds)


def _memberships(model, values, enabled):
    if not enabled:
        return None
    if not callable(getattr(model,'predict_membership',None)):
        raise CapabilityError('declared soft capability requires predict_membership')
    result = np.asarray(model.predict_membership(values.copy()))
    if result.ndim!=2 or len(result)!=len(values) or not np.isfinite(result).all():
        raise ValueError('membership output must be finite and row aligned')
    return result


def cluster_regimes(features, *, clusterer, preprocess, observations=None, fit_scope='self',
                    training_ids=None, sample_weight=None, soft=False, distances=None, metric_interpretation=None):
    observations = ObservationPolicy('observed_only') if observations is None else observations
    if not isinstance(features,RegimeFeatures):
        raise ValueError('explicit RegimeFeatures required')
    if fit_scope not in {'self','training_only'} or (fit_scope=='training_only') != (training_ids is not None):
        raise ValueError('training_only requires IDs; self fits all evaluation rows retrospectively')
    requested = features._ids if training_ids is None else list(training_ids)
    if len(set(requested))!=len(requested) or not set(requested)<=set(features._ids):
        raise ValueError('training IDs must be a unique subset')
    model, settings = _clusterer(clusterer)
    caps = settings['capabilities']
    if type(soft) is not bool or soft and not caps['soft']:
        raise CapabilityError('soft membership is unsupported')
    if sample_weight is not None and not caps['sample_weight']:
        raise CapabilityError('sample weights are unsupported')
    if distances is not None and not caps['precomputed']:
        raise CapabilityError('precomputed distances are unsupported')
    if caps['precomputed'] and distances is None:
        raise ValueError('precomputed mode requires an explicit aligned distance matrix')
    if distances is not None and preprocess is not None:
        raise CapabilityError('precomputed distances bypass preprocessing')
    if len({s['unit'] for s in features.schema})>1 and preprocess is None and metric_interpretation is None and distances is None:
        raise ValueError('mixed units require declared scaling or metric_interpretation')
    if metric_interpretation is not None:
        text_field(metric_interpretation,'metric interpretation')
    transformer, transform_spec = _preprocessor(preprocess)
    weights = None
    if sample_weight is not None:
        if not isinstance(sample_weight,pd.Series) or list(sample_weight.index)!=features._ids:
            raise ValueError('sample weights must be a Series aligned to every original row ID')
        weights = sample_weight.to_numpy(dtype=float)
        if not np.isfinite(weights).all() or np.any(weights<0) or not weights.sum()>0:
            raise ValueError('sample weights must be finite nonnegative with positive total')
    distance_values = None
    if distances is not None:
        if not isinstance(distances,pd.DataFrame) or list(distances.index)!=features._ids or list(distances)!=features._ids:
            raise ValueError('distance rows and columns must match original IDs')
        distance_values = distances.to_numpy(dtype=float,copy=True)
        if (not np.isfinite(distance_values).all() or np.any(distance_values<0)
                or not np.allclose(distance_values,distance_values.T,rtol=0,atol=1e-10)
                or not np.allclose(np.diag(distance_values),0,rtol=0,atol=1e-10)):
            raise ValueError('distances must be finite nonnegative symmetric with zero diagonal')
    statuses = _selection(features,observations,requested)
    ids = [r for r,s in zip(features._ids,statuses) if s=='assigned']
    support = _support(features,ids,observations)
    record = dict(schema=features.schema,clusterer=settings,preprocess=transform_spec,fit_scope=fit_scope,
        requested_training_ids=requested,support=support,preprocess_fit_support=None,
        source=features.describe(),metric_interpretation=metric_interpretation,
        sample_weight=None if weights is None else dict(row_ids=features._ids,values=weights),
        distance_id=None if distances is None else identity(dict(row_ids=features._ids,values=distance_values)),
        compute_status='ok',reason=None,centroids=None,diagnostics={},transformed_columns=None)
    status, reason, labels, membership = 'ok',None,[],None
    coordinates = features.values.loc[ids]
    start = perf_counter()
    if len(ids)<observations.min_points or settings['method']=='kmeans' and len(ids)<settings['parameters']['n_clusters']:
        status,reason = 'insufficient_support','too few eligible training rows'
    else:
        try:
            if transformer is not None:
                record['preprocess_fit_support'] = owned(support)
            coordinates = _transform(transformer,transform_spec,coordinates,fit=True)
            ids = list(coordinates.index)
            record['transformed_columns'] = list(coordinates.columns)
            positions = [features._ids.index(r) for r in ids]
            values = coordinates.to_numpy(dtype=float)
            if distance_values is not None:
                values = distance_values[np.ix_(positions,positions)]
            kwargs = {} if weights is None else dict(sample_weight=weights[positions].copy())
            if (len(ids)<observations.min_points or weights is not None and not weights[positions].sum()>0
                    or settings['method']=='kmeans' and len(ids)<settings['parameters']['n_clusters']):
                status,reason = 'insufficient_support','too few transformed rows or zero selected weight'
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if callable(getattr(model,'fit_predict',None)):
                        labels = model.fit_predict(values.copy(),**kwargs)
                    elif callable(getattr(model,'fit',None)):
                        model.fit(values.copy(),**kwargs)
                        labels = model.labels_
                    else:
                        raise CapabilityError('clusterer requires fit_predict or fit plus labels_')
                notices = [str(w.message) for w in caught]
                record['diagnostics']['warnings'] = notices
                limit = settings['method']=='kmeans' and model.n_iter_>=settings['parameters']['max_iter']
                if notices or limit:
                    status,reason = 'not_converged','backend warning or declared iteration limit reached'
                else:
                    membership = _memberships(model,values,soft)
                if settings['method']=='kmeans':
                    record['centroids'] = model.cluster_centers_.copy()
                    record['diagnostics'].update(inertia=float(model.inertia_),iterations=int(model.n_iter_))
        except RegimeFitError as error:
            status,reason = 'fit_failed',str(error) or 'custom fit failed'
    record.update(compute_status=status,reason=reason,support=_support(features,ids,observations))
    training_uses = [record['support']['use_status']]
    if record['preprocess_fit_support'] is not None:
        training_uses.append(record['preprocess_fit_support']['use_status'])
    record['use_status'] = 'unknown_provenance' if 'unknown_provenance' in training_uses else record['support']['use_status']
    if transformer is not None:
        record['preprocess_state'] = {k:owned(getattr(transformer,k)) for k in ('mean_','scale_','var_') if hasattr(transformer,k)}
    reference = RegimeReference(record,model,transformer)
    good_ids = ids if status=='ok' else []
    return _result(features,statuses,labels if status=='ok' else [],good_ids,reference,coordinates,observations,
        status=status,reason=reason,membership=membership,distances=distance_values,cost_seconds=perf_counter()-start)


def assign_regimes(features, *, reference, observations=None, soft=False):
    if not isinstance(features,RegimeFeatures) or not isinstance(reference,RegimeReference):
        raise ValueError('assignment requires RegimeFeatures and RegimeReference')
    frozen = reference.describe()
    if not frozen['clusterer']['capabilities']['assignment'] or frozen['distance_id'] is not None:
        raise CapabilityError('this fitted method has no native future assignment')
    if identity(features.schema)!=identity(frozen['schema']):
        raise ValueError('evaluation feature schema/order/units/definitions must match training')
    if type(soft) is not bool or soft and not frozen['clusterer']['capabilities']['soft']:
        raise CapabilityError('soft membership is unsupported')
    observations = ObservationPolicy('observed_only') if observations is None else observations
    statuses = _selection(features,observations)
    ids = [r for r,s in zip(features._ids,statuses) if s=='assigned']
    coordinates = features.values.loc[ids]
    status,reason = frozen['compute_status'],frozen['reason']
    model, transformer = reference.fitted_model,reference.fitted_preprocess
    labels, membership = [],None
    start = perf_counter()
    if status=='ok' and len(ids)<observations.min_points:
        status,reason = 'insufficient_support','too few eligible evaluation rows'
    if status=='ok':
        coordinates = _transform(transformer,frozen['preprocess'],coordinates,fit=False)
        ids = list(coordinates.index)
        if list(coordinates.columns)!=frozen['transformed_columns']:
            raise ValueError('transformed evaluation schema differs from training')
        if len(ids)<observations.min_points:
            status,reason = 'insufficient_support','too few transformed evaluation rows'
        else:
            if not callable(getattr(model,'predict',None)):
                raise CapabilityError('declared assignment requires genuine predict')
            try:
                labels = model.predict(coordinates.to_numpy(dtype=float,copy=True))
                membership = _memberships(model,coordinates.to_numpy(dtype=float),soft)
            except RegimeFitError as error:
                status,reason = 'fit_failed',str(error) or 'custom assignment failed'
    return _result(features,statuses,labels if status=='ok' else [],ids if status=='ok' else [],reference,coordinates,
        observations,status=status,reason=reason,membership=membership,cost_seconds=perf_counter()-start)


def _adjusted_rand(first, second):
    # Pair-count formula; label names and their permutations have no meaning.
    n = len(first)
    if n<2:
        return None
    def counts(items):
        result = {}
        for item in items:
            result[item] = result.get(item,0)+1
        return result.values()
    pairs = lambda c: sum(k*(k-1)/2 for k in c)
    together = pairs(counts(zip(first,second)))
    a,b,total = pairs(counts(first)),pairs(counts(second)),n*(n-1)/2
    expected = a*b/total
    maximum = (a+b)/2
    return 1.0 if maximum==expected else float((together-expected)/(maximum-expected))


def compare_regime_models(candidates, *, noise_policy):
    """Compare a declared finite list of (name, RegimeResult), without selecting a winner."""
    if not isinstance(candidates,(list,tuple)) or not candidates:
        raise ValueError('explicit finite nonempty candidate list required')
    if noise_policy not in {'include','exclude'}:
        raise ValueError('noise_policy must be include or exclude')
    names = [c[0] for c in candidates]
    if len(set(names))!=len(names):
        raise ValueError('candidate names must be unique')
    summaries, selected = [],{}
    for name,result in candidates:
        text_field(name,'candidate name')
        if not isinstance(result,RegimeResult):
            raise ValueError('candidates must contain already computed RegimeResult objects')
        record = result.describe()
        rows = result.rows
        keep = rows.status.eq('assigned') | (rows.status.eq('noise') if noise_policy=='include' else False)
        rows = rows.loc[keep]
        ids = list(rows.row_id)
        labels = [('noise',) if r.status=='noise' else ('label',r.native_label) for r in rows.itertuples()]
        selected[name] = (rows,labels)
        n,k = len(labels),len(set(labels))
        separation,why = None,'requires between two and n-1 clusters'
        metric = record['training']['clusterer']['parameters'].get('metric','euclidean')
        if record['compute_status']!='ok':
            why = 'candidate unavailable'
        elif 1<k<n:
            if result._distances is not None:
                all_ids = [r['row_id'] for r in record['source']['rows']]
                positions = [all_ids.index(r) for r in ids]
                d = result._distances[np.ix_(positions,positions)]
            elif metric=='euclidean':
                x = result._coordinates.loc[ids].to_numpy(dtype=float)
                d = cdist(x,x)
            else:
                d = None
                why = 'separation unavailable for this metric'
            if d is not None:
                scores = []
                for i,label in enumerate(labels):
                    same = [j for j,v in enumerate(labels) if v==label and i!=j]
                    if not same:
                        scores.append(0.0)
                        continue
                    a = float(d[i,same].mean())
                    b = min(float(d[i,[j for j,v in enumerate(labels) if v==other]].mean()) for other in set(labels)-{label})
                    scores.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
                separation,why = float(np.mean(scores)),None
        summaries.append(dict(candidate=name,result_id=result.identity,compute_status=record['compute_status'],
            reason=record['reason'],use_status=record['use_status'],reference_id=record['reference_id'],
            cluster_sizes=record['cluster_sizes'],cluster_count=record['cluster_count'],
            denominator=record['denominator'],evaluated=len(ids),noise=record['noise'],excluded=record['excluded'],failed=record['failed'],
            fractions=record['fractions'],silhouette=separation,silhouette_reason=why,cost_seconds=record['cost_seconds']))
    agreements = []
    for i,(a,first) in enumerate(candidates):
        for b,second in candidates[i+1:]:
            ar,al = selected[a]; br,bl = selected[b]
            left,right = dict(zip(ar.row_id,al)),dict(zip(br.row_id,bl))
            ids = [r for r in left if r in right]
            same_schema = identity(first.describe()['schema'])==identity(second.describe()['schema'])
            geometry_a = {r.row_id:(r.start,r.end) for r in ar.itertuples()}
            geometry_b = {r.row_id:(r.start,r.end) for r in br.itertuples()}
            compatible = same_schema and all(geometry_a[r]==geometry_b[r] for r in ids)
            available = first.describe()['compute_status']==second.describe()['compute_status']=='ok'
            value = _adjusted_rand([left[r] for r in ids],[right[r] for r in ids]) if compatible and available else None
            agreements.append(dict(first=a,second=b,row_ids=ids,denominator=len(ids),adjusted_rand=value,
                compute_status='incomparable' if not compatible else 'ok' if value is not None else 'unavailable',
                reason=None if value is not None else 'incompatible schema/support or insufficient available common IDs'))
    return owned(dict(candidates=pd.DataFrame(summaries),agreements=pd.DataFrame(agreements),noise_policy=noise_policy,
        interpretation='Descriptive partitions on dependent temporal rows; no winner or independent-replicate confidence.'))


def summarize_regime_timeline(result, *, max_gap, noise_policy):
    """Retain barriers and noise; episode duration is union coverage, never a row count."""
    if not isinstance(result,RegimeResult) or noise_policy not in {'include','barrier'}:
        raise ValueError('RegimeResult and explicit include/barrier noise policy required')
    gap = positive_duration(max_gap,'max_gap')
    rows = result.rows.to_dict('records')
    previous_start = None
    seen = set()
    for row in rows:
        start,end = row['start'],row['end']
        if row['row_id'] in seen or previous_start is not None and start<=previous_start or end<start:
            raise ValueError('timeline requires unique IDs and ordered nonreversed support')
        seen.add(row['row_id']); previous_start = start
    episodes, transitions = [],[]
    active, previous = None,None
    for row in rows:
        if row['status'] not in {'assigned','noise'} or row['status']=='noise' and noise_policy=='barrier':
            active,previous = None,None
            continue
        label = (row['status'],row['native_label'])
        coverage_end = max(end for _,end in active['intervals']) if active is not None else None
        continuous = previous is not None and row['start']-coverage_end<=gap
        if active is None or not continuous or active['key']!=label:
            if previous is not None and continuous:
                transitions.append(dict(from_row_id=previous['row_id'],to_row_id=row['row_id'],
                    from_label=previous['native_label'],to_label=row['native_label'],
                    lower=previous['start'],upper=max(previous['end'],row['end']),
                    meaning='transition bounded by feature supports, not a precise change point'))
            active = dict(key=label,row_ids=[],intervals=[],native_label=row['native_label'],status=row['status'])
            episodes.append(active)
        active['row_ids'].append(row['row_id'])
        active['intervals'].append((row['start'],row['end']))
        previous = row
    for episode in episodes:
        intervals = episode.pop('intervals')
        left,right = intervals[0]
        covered = 0
        for start,end in intervals[1:]:
            if start>right:
                covered += right.value-left.value
                left,right = start,end
            else:
                right = max(right,end)
        covered += right.value-left.value
        episode.pop('key')
        episode.update(start=intervals[0][0],end=max(e for _,e in intervals),covered_seconds=covered/1e9,
            elapsed_span_seconds=(max(e for _,e in intervals).value-intervals[0][0].value)/1e9)
    return dict(rows=result.rows,episodes=_records_frame(episodes),transitions=_records_frame(transitions),
        source_result_id=result.identity,max_gap=gap,noise_policy=noise_policy,
        warning='No smoothing; noise is distinct from exclusions; overlap is dependent support.')

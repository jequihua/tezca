"""Fixed M004 synthetic bank; the two supplied draws are not imputation methods."""
import json

import numpy as np
import pandas as pd

from tezca.data import AnalysisSeries
from tezca.irregular import IrregularSpec
from tezca.sensitivity import spectral_sensitivity, summarize_draws
from tezca.spectral import WelchSpec
from tezca.spectral_summary import ModelSpec


START = pd.Timestamp("2000-02-01",tz="UTC")


def declared(values, *, start=START, source_id="fixed-sensitivity-evaluation"):
    values = np.asarray(values)
    if values.ndim==1:
        values = values[:,None]
    names = ["x","y","z"][:values.shape[1]]
    return AnalysisSeries(values,variables=names,time=pd.date_range(start,periods=len(values),freq="h"),
        metadata={v:dict(unit="m",representation="raw") for v in names},
        source=dict(id=source_id,segment="synthetic"),clock=dict(mode="regular",cadence="1h"),
        observed=np.isfinite(values))


def masked_copy(data,mask):
    values = data.values
    values.values[mask] = np.nan
    qualification = data.qualification
    qualification["artificial"] = np.asarray(mask,dtype=bool)
    return AnalysisSeries(values,metadata=data.metadata,source=data.source,
        clock=dict(mode="regular",cadence="1h"),observed=np.isfinite(values),qualification=qualification)


def supplied_draw(masked,draw,value):
    values = masked.values
    missing = values.isna().to_numpy()
    values.values[missing] = value
    contribution = masked.contribution
    contribution[missing] = "imputed"
    source = dict(masked.source,imputation=dict(input_id=masked.identity,draw_id=draw,
        method_id="externally_supplied_constant_fixture",plan_id="M004-fixed-74002"))
    return AnalysisSeries(values,metadata=masked.metadata,source=source,
        clock=dict(mode="regular",cadence="1h"),contribution=contribution,qualification=masked.qualification)


def settings(end=672):
    frequency = np.arange(2,31)/(96*3600)
    return dict(welch=WelchSpec(96,overlap=48,taper="boxcar",detrend=False),
        irregular=IrregularSpec(interval=[START,START+pd.Timedelta(hours=end)],segment="96h",hop="48h",anchor=START,
            frequency_hz=frequency,requested_band_hz=[frequency[0],frequency[-1]],f_ref_hz=1/3600,min_span="48h"),
        model=ModelSpec(interval_hz=[frequency[0],frequency[-1]],candidates_hz=[]),
        fit_frequency_hz=frequency,min_cycles=1)


def fixed_bank():
    n = np.arange(-672,672)
    noise = np.random.default_rng(74002).normal(size=(1344,3))
    values = np.column_stack([np.sin(2*np.pi*n/24)+.3*noise[:,0],
        .7*np.sin(2*np.pi*(n-3)/24)+.3*noise[:,1],.3*noise[:,2]])
    training = declared(values[:672],start=START-pd.Timedelta("672h"),source_id="fixed-sensitivity-training")
    complete = declared(values[672:])
    cases = []
    designs = [("none",[]),("scattered24",24*np.arange(24)+12),("block24",np.arange(120,144)),
        ("early72",np.arange(120,192)),("late72",np.arange(480,552))]
    for name,positions in designs:
        for scope in (["none"] if name=="none" else ["partial","shared"]):
            mask = np.zeros((672,3),dtype=bool)
            mask[np.asarray(positions,dtype=int),:1 if scope=="partial" else 3] = True
            masked = masked_copy(complete,mask)
            draws = {"plus":supplied_draw(masked,"plus",.25),"minus":supplied_draw(masked,"minus",-.25)}
            cases.append((name+"_"+scope,masked,draws))
    return training,complete,cases


def run_bank():
    training,complete,cases = fixed_bank()
    reports = {name:spectral_sensitivity(complete,masked,draws,variables=["x","y","z"],**settings())
               for name,masked,draws in cases}
    return training,complete,cases,reports


def compact(reports):
    return {name:dict(routes=len(report.routes),contrasts=len(report.contrasts),
        route_statuses=report.routes.compute_status.value_counts().to_dict(),
        contrast_statuses=report.contrasts.compute_status.value_counts().to_dict(),
        mean_draw_beta=summarize_draws(report)[["variable","mean","denominator"]].to_dict("records"))
        for name,report in reports.items()}


if __name__ == "__main__":
    print(json.dumps(compact(run_bank()[3]),sort_keys=True))

"""Compute before-sort certificates on the reproduced fixed query outcomes."""
from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
import design
import engine
from crossing_bounds import query_certificate as cost_range_certificate
from precision import query_certificate

ROOT = Path(__file__).resolve().parents[1]


def main():
    primary = ROOT/'run/primary'
    with tempfile.TemporaryDirectory(prefix='precision-loader-') as temp:
        design.OUT = Path(temp)
        domains,_,_,_ = design.load_frames()
    fits = {(r['domain'],r['run']):r for r in json.loads((primary/'fitted-models.json').read_text())}
    catalog = pd.read_csv(primary/'registered-query-catalog.csv',dtype={'map':str,'query_id':str,'mask_hex':str})
    outcomes = pd.read_csv(primary/'query-results.csv',dtype={'map':str,'query_id':str})
    geometries, crossing_rows, precision_rows = {},[],[]
    for row in outcomes[outcomes.method=='transport_scan'].itertuples():
        key = row.domain,row.map
        if key not in geometries:
            geometries[key] = engine.geometry(domains[row.domain][row.map])
        geo = geometries[key]
        record = catalog[(catalog.domain==row.domain)&(catalog['map']==row.map)&(catalog.query_id==row.query_id)].iloc[0]
        q = np.unpackbits(np.frombuffer(bytes.fromhex(record.mask_hex),dtype=np.uint8),bitorder='little')[:int(record.n)]
        tau = fits[row.domain,row.run]['radii']['transport_scan']
        before, after = cost_range_certificate(geo,q,tau),query_certificate(geo,q,tau)
        width = before['coarse_width']; observed = 1.-row.width_ratio
        lo,hi,new_hi = before['narrowing_lower']/width,before['narrowing_upper']/width,after['narrowing_upper']/width
        assert lo<=observed+1e-10 and observed<=new_hi+1e-10 and new_hi<=hi+1e-10
        useful,broad = lo>.05+1e-10,hi<.05-1e-10
        crossing_rows.append(dict(domain=row.domain,run=row.run,map=row.map,query_id=row.query_id,
            tau=tau,coarse_width_fraction=width,saturation_budget=before['saturation_budget'],
            plus_budget=before['plus']['saturation_budget'],minus_budget=before['minus']['saturation_budget'],
            ratio_tau_to_saturation=tau/before['saturation_budget'] if before['saturation_budget']>0 else float('inf'),
            observed_narrowing=observed,certified_narrowing_lower=lo,certified_narrowing_upper=hi,
            exact_saturation=tau>=before['saturation_budget'],useful_recorded=row.width_ratio<.95,
            certified_useful=useful,certified_broad=broad,requires_sort_for_classification=not(useful or broad)))
        useful,broad = lo>.05+1e-10,new_hi<.05-1e-10
        precision_rows.append(dict(domain=row.domain,run=row.run,map=row.map,query_id=row.query_id,
            tau=tau,coarse_width_fraction=width,narrowing_lower=lo,narrowing_observed=observed,narrowing_upper=new_hi,
            previous_narrowing_upper=hi,upper_strictly_tighter=new_hi<hi-1e-10,
            exact_saturation=tau>=after['saturation_budget'],certified_useful=useful,certified_broad=broad,
            requires_sort_for_classification=not(useful or broad)))
    assert len(precision_rows)==1627
    for name,rows in [('crossing',crossing_rows),('precision',precision_rows)]:
        destination=ROOT/'run'/name;destination.mkdir(exist_ok=False)
        data=pd.DataFrame(rows);data.to_csv(destination/'query-diagnostics.csv',index=False)
        summaries=[]
        for (domain,run),g in data.groupby(['domain','run'],sort=False):
            bymap=g.groupby('map')
            if name=='precision':
                summary=dict(domain=domain,run=run,maps=g['map'].nunique(),queries=len(g),
                    **{k:int(g[k].sum()) for k in ['exact_saturation','upper_strictly_tighter','certified_useful','certified_broad','requires_sort_for_classification']},
                    **{'map_macro_'+k:float(bymap[k].mean().mean()) for k in ['narrowing_lower','narrowing_observed','narrowing_upper','previous_narrowing_upper']})
            else:
                summary=dict(domain=domain,run=run,queries=len(g),maps=g['map'].nunique(),
                    exact_saturated_queries=int(g.exact_saturation.sum()),recorded_useful_queries=int(g.useful_recorded.sum()),
                    certified_useful_queries=int(g.certified_useful.sum()),certified_broad_queries=int(g.certified_broad.sum()),
                    requires_sort_queries=int(g.requires_sort_for_classification.sum()),
                    map_macro_lower_narrowing=float(bymap.certified_narrowing_lower.mean().mean()),
                    map_macro_observed_narrowing=float(bymap.observed_narrowing.mean().mean()),
                    map_macro_upper_narrowing=float(bymap.certified_narrowing_upper.mean().mean()))
            summaries.append(summary)
        pd.DataFrame(summaries).to_csv(destination/'summary.csv',index=False)
    print('Checked',len(precision_rows),'queries; unchanged endpoints and tighter feasible-plan upper bound.')


if __name__=='__main__':
    main()

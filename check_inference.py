"""Structural checks of the finite Bayesian calculation, using archived runs."""
from pathlib import Path
import argparse
import json
import numpy as np
from inference_design import ROOT, CONFIG, posterior, normalized, write_json


def run(results):
    lib=dict(np.load(results/'library.npz'))
    a3=lib['parameters'][:,4];nodes=np.unique(a3)
    prior=lib['prior'];marginal=np.array([prior[a3==x].sum() for x in nodes])
    checks={'support_size':len(prior),'prior_weight_sum_error':float(abs(prior.sum()-1)),
            'third_pulse_initial_response_max':float(abs(lib['basis'][:,0,2]).max()),
            'source3_prior_nodes_mg_L':nodes.tolist(),'source3_prior_probabilities':marginal.tolist(),
            'case_checks':[]}
    groups = [results] + [results/group for group in ('development', 'heldout')]
    for group in groups:
        for path in sorted(group.glob('case_*.json')):
            case=json.loads(path.read_text());w=posterior(lib,np.array(case['initial_data']))
            new=np.array([w[a3==x].sum() for x in nodes]);error=float(abs(new-marginal).max())
            checks['case_checks'].append({'case_seed':case['seed'],'source3_marginal_max_change':error,
                                          'posterior_sum_error':float(abs(w.sum()-1))})
            assert error<1e-12, 'Past data spuriously update an independent future pulse'
    assert checks['third_pulse_initial_response_max']==0
    assert checks['prior_weight_sum_error']<1e-12
    checks['maximum_source3_marginal_change']=max(r['source3_marginal_max_change'] for r in checks['case_checks'])
    # Two independent likelihood factors must compose irrespective of order.
    rng=np.random.default_rng(762)
    ll1=-rng.random(len(prior))*3;ll2=-rng.random(len(prior))*2
    together=normalized(np.log(prior)+ll1+ll2)
    sequential=normalized(np.log(normalized(np.log(prior)+ll1))+ll2)
    checks['likelihood_composition_max_error']=float(abs(together-sequential).max())
    assert checks['likelihood_composition_max_error']<1e-12
    checks['status']='passed'
    write_json(results/'inference_checks.json',checks)
    print(json.dumps({k:v for k,v in checks.items() if k!='case_checks'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    run(p.parse_args().results)

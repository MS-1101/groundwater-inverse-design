"""Finite-ensemble Bayesian inversion and batch monitoring design.

The ensemble is a documented quadrature approximation to a continuous prior.
The inner posterior is summed exactly on this finite support; outer predictive
measurements are Monte Carlo. Design and validation draws use different seeds.
"""
from __future__ import annotations
import argparse
from itertools import product, combinations
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc
from physics import solve_basis, sample_cells

ROOT=Path(__file__).resolve().parent
CANDIDATES=np.array([(i,j) for i in (4,6,8,10,12,14) for j in (3,5,7,9)])
INITIAL_CONC=np.array([(4,6),(8,6),(10,6)])
INITIAL_HEAD=np.array([(6,3),(9,10)])
RECEPTOR=(10,6)
CONFIG={
    'status':'Synthetic benchmark experiment; no field observations',
    'grid_refinement':4,'dt_days':2.,'times_days':[180.,365.,545.],
    'k_quadrature_order':3,'source_quadrature_order':5,'prior_rule':'tensor Gauss-Hermite; independent source axes',
    'log_k_standard_deviation':[.35,.7], 'source_median_mg_L':57.87,
    'source_log_standard_deviation':.45,'pulse_edges_days':[0.,120.,240.,360.],
    'initial_concentration_cells':INITIAL_CONC.tolist(),'initial_head_cells':INITIAL_HEAD.tolist(),
    'candidate_cells':CANDIDATES.tolist(),'receptor_cell':list(RECEPTOR),
    'concentration_noise_sd_mg_L':.08,'head_noise_sd_m':1.5,
    'future_threshold_mg_L':.6,'false_negative_cost':9.,'false_positive_cost':1.,
    'batch_size':4,'design_outer_draws':256,'validation_outer_draws':4096,
    'development_case_seeds':[1101,1102,1103],
    'heldout_case_seeds':list(range(3001,3013)),
    'selection':'greedy batch with common random numbers; no sequential observations within batch',
    'evaluation':'independent outer Monte Carlo; exact posterior summation on finite support',
}


def write_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))


def hash_config(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def normalized(logw):
    return np.exp(logw-logsumexp(logw,axis=0,keepdims=True))


def weighted_quantile(x,w,quantile):
    order=np.argsort(x)
    index=min(int(np.searchsorted(np.cumsum(w[order]),quantile,side="left")),len(order)-1)
    return float(x[order[index]])


def build_library(cfg=CONFIG,order=None,refinement=None,dt=None):
    started=time.perf_counter()
    order=cfg['k_quadrature_order'] if order is None else order
    refinement=cfg['grid_refinement'] if refinement is None else refinement
    dt=cfg['dt_days'] if dt is None else dt
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    axes=[np.sqrt(2)*nodes*s for s in cfg['log_k_standard_deviation']]
    weights=weights/np.sqrt(np.pi)
    an,aw=np.polynomial.hermite.hermgauss(cfg['source_quadrature_order'])
    source_nodes=cfg['source_median_mg_L']*np.exp(cfg['source_log_standard_deviation']*np.sqrt(2)*an)
    aw=aw/np.sqrt(np.pi)
    source_indices=np.array(list(product(range(len(an)),repeat=3)))
    amps=source_nodes[source_indices]
    source_weights=np.prod(aw[source_indices],axis=1)
    prior=[];params=[];initial=[];future=[];candidates=[];allbasis=[];allhead=[];ks=[];checks=[]
    for i,j in product(range(order),repeat=2):
        logk=(axes[0][i],axes[1][j]);ks.append(logk)
        sim=solve_basis(logk,times=cfg['times_days'],dt_days=dt,refinement=refinement)
        basis=sim['basis'];head=sim['head'];allbasis.append(basis);allhead.append(head)
        b0=sample_cells(basis[0],INITIAL_CONC,refinement)
        bc=sample_cells(basis[1],CANDIDATES,refinement)
        bq=sample_cells(basis[2],[RECEPTOR],refinement)[:,0]
        hc=sample_cells(head,INITIAL_HEAD,refinement)
        initial.append(np.column_stack([amps@b0,np.repeat(hc[None,:],len(amps),axis=0)]))
        candidates.append(amps@bc);future.append(amps@bq)
        params.append(np.column_stack([np.repeat(np.array(logk)[None,:],len(amps),axis=0),amps]))
        prior.extend(weights[i]*weights[j]*source_weights)
        checks.append({'log_k':list(logk),'mass_balance_relative':sim['mass_balance_relative'],
                       'flow_residual_m3_day':sim['flow_residual_m3_day'],'elapsed_s':sim['elapsed_s']})
    return {'prior':np.array(prior),'parameters':np.concatenate(params),
            'initial_predictions':np.concatenate(initial),'candidate_predictions':np.concatenate(candidates),
            'future_concentration':np.concatenate(future),'basis':np.array(allbasis),'heads':np.array(allhead),
            'log_k_nodes':np.array(ks),'source_amplitudes':amps,
            'build_seconds':np.array(time.perf_counter()-started),
            'mass_balance_max':np.array(max(c['mass_balance_relative'] for c in checks)),
            'configuration_sha256':np.array(hash_config(cfg)),
            'library_refinement':np.array(refinement),'library_order':np.array(order),'library_dt_days':np.array(dt)}


def initial_sigma(cfg=CONFIG):
    return np.array([cfg['concentration_noise_sd_mg_L']]*3+[cfg['head_noise_sd_m']]*2)


def posterior(library,data,cfg=CONFIG):
    residual=(library['initial_predictions']-data)/initial_sigma(cfg)
    return normalized(np.log(library['prior'])-.5*np.sum(residual**2,axis=1))


def posterior_summary(library,w,cfg=CONFIG):
    q=library['future_concentration'];p=float(np.clip(w@(q>cfg['future_threshold_mg_L']),0,1))
    fn,fp=cfg['false_negative_cost'],cfg['false_positive_cost']
    return {'future_mean_mg_L':float(w@q),'future_sd_mg_L':float(np.sqrt(max(w@q**2-(w@q)**2,0))),
            'future_q025_mg_L':weighted_quantile(q,w,.025),'future_q975_mg_L':weighted_quantile(q,w,.975),
            'exceedance_probability':p,'action_strengthen_monitoring':bool(fn*p>fp*(1-p)),
            'bayes_risk':min(fn*p,fp*(1-p)), 'effective_support_size':float(1/(w@w))}


class PredictiveExperiment:
    """Common predictive draws; candidate-wise log-likelihood caches."""
    def __init__(self,library,w,draws,seed,cfg=CONFIG):
        self.library,self.w,self.cfg=library,w,cfg
        self.q=library['future_concentration'];self.event=self.q>cfg['future_threshold_mg_L']
        rng=np.random.default_rng(seed)
        self.truth_indices=rng.choice(len(w),draws,p=w)
        means=library['candidate_predictions'];sigma=cfg['concentration_noise_sd_mg_L']
        self.y=means[self.truth_indices]+sigma*rng.normal(size=(draws,len(CANDIDATES)))
        self.logw=np.log(np.maximum(w,1e-300))[:,None]
        self.cache=[-.5*((means[:,j,None]-self.y[None,:,j])/sigma)**2 for j in range(len(CANDIDATES))]

    def score(self,selected,details=False):
        ll=sum((self.cache[j] for j in selected),start=np.zeros_like(self.logw))
        logpost=self.logw+ll;logpost-=logsumexp(logpost,axis=0,keepdims=True)
        weights=np.exp(logpost)
        p=np.clip(self.event@weights,0,1);m=self.q@weights
        var=np.maximum((self.q**2)@weights-m**2,0)
        risk=np.minimum(self.cfg['false_negative_cost']*p,self.cfg['false_positive_cost']*(1-p))
        information=np.sum(weights*(logpost-self.logw),axis=0)
        out={'decision_risk':float(risk.mean()),'predictive_variance':float(var.mean()),
             'joint_information_nats':float(information.mean())}
        if details:
            truth_q=self.q[self.truth_indices];truth_event=truth_q>self.cfg['future_threshold_mg_L']
            action=self.cfg['false_negative_cost']*p>self.cfg['false_positive_cost']*(1-p)
            losses=np.where(truth_event & ~action,self.cfg['false_negative_cost'],
                            np.where(~truth_event & action,self.cfg['false_positive_cost'],0.))
            out.update({'risk_standard_error':float(risk.std(ddof=1)/np.sqrt(len(risk))),
                        'predictive_rmse_mg_L':float(np.sqrt(np.mean((m-truth_q)**2))),
                        'realized_loss_mean':float(losses.mean()),
                        'false_negative_count':int(np.sum(truth_event & ~action)),
                        'false_positive_count':int(np.sum(~truth_event & action)),
                        'true_exceedance_count':int(truth_event.sum()),'draws':len(risk),
                        'posterior_effective_support_mean':float(np.mean(1/np.sum(weights**2,axis=0)))})
            return out,{'risk_by_draw':risk,'variance_by_draw':var,'predicted_probability':p,
                        'realized_loss_by_draw':losses,'truth_indices':self.truth_indices}
        return out


def select_greedy(experiment,method,budget=4,pool=None):
    selected=[];trace=[];started=time.perf_counter()
    pool=list(range(len(CANDIDATES))) if pool is None else list(pool)
    for depth in range(budget):
        values=[]
        for j in pool:
            if j in selected:continue
            metrics=experiment.score(selected+[j])
            criterion=-metrics[method] if method=='joint_information_nats' else metrics[method]
            values.append((criterion,j,metrics))
        value,j,metrics=min(values,key=lambda v:(v[0],v[1]));selected.append(j)
        trace.append({'depth':depth+1,'selected_index':j,'metrics':metrics,
                      'candidate_scores':[{'index':v[1],'criterion':v[0]} for v in values]})
    return selected,trace,time.perf_counter()-started


def select_coverage(budget=4,pool=None):
    selected=[];pool=list(range(len(CANDIDATES))) if pool is None else list(pool)
    fixed=INITIAL_CONC.copy()
    for _ in range(budget):
        points=np.vstack([fixed,CANDIDATES[selected]]) if selected else fixed
        dist=((CANDIDATES[:,None,:]-points[None,:,:])**2).sum(axis=2).min(axis=1)
        j=min([j for j in pool if j not in selected],key=lambda j:(-dist[j],j));selected.append(j)
    return selected


def run_case(library,seed,cfg=CONFIG,draws=None,validation_draws=None):
    started=time.perf_counter();rng=np.random.default_rng(seed)
    truth=int(rng.choice(len(library['prior']),p=library['prior']))
    initial=library['initial_predictions'][truth]+initial_sigma(cfg)*rng.normal(size=5)
    w=posterior(library,initial,cfg);before=posterior_summary(library,w,cfg)
    train=PredictiveExperiment(library,w,draws or cfg['design_outer_draws'],seed+100000,cfg)
    plans={'coverage':(select_coverage(cfg['batch_size']),[],0.)}
    for name,criterion in [('parameter_information','joint_information_nats'),('prediction_variance','predictive_variance'),('decision_loss','decision_risk')]:
        plans[name]=select_greedy(train,criterion,cfg['batch_size'])
    validation=PredictiveExperiment(library,w,validation_draws or cfg['validation_outer_draws'],seed+200000,cfg)
    rows=[];arrays={'initial_data':initial,'initial_posterior':w,'truth_index':np.array(truth)};traces={}
    realized_y=library['candidate_predictions'][truth]+cfg['concentration_noise_sd_mg_L']*rng.normal(size=24)
    for method,(selected,trace,elapsed) in plans.items():
        metrics,detail=validation.score(selected,True)
        # A separate illustrative observed update, generated from this case's truth.
        residual=(library['candidate_predictions'][:,selected]-realized_y[selected])/cfg['concentration_noise_sd_mg_L']
        wafter=normalized(np.log(np.maximum(w,1e-300))-.5*np.sum(residual**2,axis=1))
        after=posterior_summary(library,wafter,cfg)
        rows.append({'case_seed':seed,'method':method,'selected_indices':selected,
                     'selected_wells':[f'W{j+1:02d}' for j in selected],
                     'before_risk':before['bayes_risk'],'before_effective_support':before['effective_support_size'],
                     'selection_seconds':elapsed,**metrics,'illustrative_after':after})
        arrays[method+'_posterior_after']=wafter
        for k,v in detail.items():arrays[method+'_'+k]=v
        traces[method]=trace
    arrays['illustrative_new_data_all_candidates']=realized_y
    return {'seed':seed,'truth_index':truth,'initial_data':initial.tolist(),'before':before,
            'rows':rows,'selection_traces':traces,'case_total_seconds':time.perf_counter()-started},arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results/main')
    p.add_argument('--cases',default='development')
    p.add_argument('--library',type=Path,default=ROOT/'results/library.npz')
    p.add_argument('--freeze',type=Path,default=ROOT/'protocol.json')
    args=p.parse_args();cfg=dict(CONFIG)
    args.out.mkdir(parents=True,exist_ok=True)
    seeds=cfg['development_case_seeds'] if args.cases=='development' else cfg['heldout_case_seeds'] if args.cases=='heldout' else list(map(int,args.cases.split(',')))
    stamp={'config':cfg,'config_sha256':hash_config(cfg),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                          'hardware':'CPU; BLAS thread limits set by launch command'},
           'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['physics.py','inference_design.py']}}
    if args.freeze.exists():assert json.loads(args.freeze.read_text())['config_sha256']==stamp['config_sha256'],'Protocol changed'
    else:
        assert not any(s in cfg['heldout_case_seeds'] for s in seeds),'Freeze on development before heldout'
        write_json(args.freeze,stamp)
    write_json(args.out/'manifest.json',stamp)
    if args.library.exists():
        library=dict(np.load(args.library))
        assert str(library['configuration_sha256']) == hash_config(cfg),'Library/config mismatch'
    else:
        library=build_library(cfg);args.library.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.library,**library)
    rows=[]
    for seed in seeds:
        print(json.dumps({'event':'case_start','seed':seed,'support_size':len(library['prior'])}),flush=True)
        case,arrays=run_case(library,seed,cfg)
        write_json(args.out/f'case_{seed}.json',case);np.savez_compressed(args.out/f'case_{seed}.npz',**arrays)
        rows.extend(case['rows']);write_json(args.out/'results.json',rows)
        print(json.dumps({'event':'case_complete','seed':seed,'seconds':case['case_total_seconds'],
                          'before_risk':case['before']['bayes_risk'],
                          'risk':{r['method']:r['decision_risk'] for r in case['rows']}}),flush=True)
    print(json.dumps({'event':'complete','rows':len(rows),'out':str(args.out)}),flush=True)


if __name__=='__main__':main()

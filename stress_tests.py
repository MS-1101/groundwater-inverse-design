"""Exploratory supplements: finite-pool enumeration and continuous-prior stress.

These do not replace or tune the frozen main experiment. Truth is generated
from independently sampled continuous parameters; no field measurements occur.
"""
from pathlib import Path
from itertools import combinations
import argparse,csv,hashlib,json,platform,time
import numpy as np
from inference_design import (CONFIG,CANDIDATES,INITIAL_CONC,INITIAL_HEAD,RECEPTOR,
    PredictiveExperiment,posterior,posterior_summary,initial_sigma,normalized,
    select_greedy,select_coverage,write_json,hash_config)
from physics import solve_basis,sample_cells

ROOT=Path(__file__).resolve().parent
METHODS=[('parameter_information','joint_information_nats'),('prediction_variance','predictive_variance'),('decision_loss','decision_risk')]

def enumeration(library,out,development_case=ROOT/'results/case_1102.npz'):
    development_case=Path(development_case).resolve()
    case=np.load(development_case)
    w=case['initial_posterior'];pool=list(range(0,24,2))
    train=PredictiveExperiment(library,w,256,1102+100000)
    start=time.perf_counter();scores=[]
    for selected in combinations(pool,3):
        metrics=train.score(selected);scores.append(dict(selected_indices=list(selected),**metrics))
    best=min(scores,key=lambda x:(x['decision_risk'],x['selected_indices']))
    enum_time=time.perf_counter()-start
    greedy,trace,greedy_time=select_greedy(train,'decision_risk',3,pool)
    greedy_train=train.score(greedy)
    validation=PredictiveExperiment(library,w,4096,1102+700000)
    evals={};details={}
    for label,selected in [('enumerated_best',best['selected_indices']),('greedy',greedy)]:
        metrics,detail=validation.score(selected,True)
        evals[label]=dict(selected_indices=selected,selected_wells=[f'W{i+1:02d}' for i in selected],**metrics)
        details[label]=detail
    diff=details['greedy']['risk_by_draw']-details['enumerated_best']['risk_by_draw']
    result=dict(status='Exploratory enumeration on a fixed Monte Carlo objective, not exact population-risk global optimization',
                development_seed=1102,development_case=str(development_case),
                development_case_sha256=hashlib.sha256(development_case.read_bytes()).hexdigest(),
                pool_indices=pool,budget=3,combinations_evaluated=len(scores),
                design_outer_draws=256,validation_outer_draws=4096,
                design_seed=101102,validation_seed=701102,
                enumerated_best_training=best,greedy_training=dict(selected_indices=greedy,**greedy_train),
                enumeration_seconds=enum_time,greedy_seconds=greedy_time,
                evaluation=evals,greedy_minus_enumerated_validation_risk=float(diff.mean()),
                paired_standard_error=float(diff.std(ddof=1)/np.sqrt(len(diff))),
                all_training_scores=scores,greedy_trace=trace)
    write_json(out/'enum.json',result)
    np.savez_compressed(out/'enum_raw.npz',initial_posterior=w,
                        enumerated_risk=details['enumerated_best']['risk_by_draw'],
                        greedy_risk=details['greedy']['risk_by_draw'],
                        validation_truth_indices=details['greedy']['truth_indices'])
    print(json.dumps({'event':'enumeration_complete','train_best':best['decision_risk'],'train_greedy':greedy_train['decision_risk'],
                      'validation_difference':float(diff.mean()),'paired_se':result['paired_standard_error']}),flush=True)
    return result

def continuous_case(library,seed,group,refinement,dt_days):
    start=time.perf_counter();rng=np.random.default_rng(seed)
    logk=rng.normal(size=2)*np.array(CONFIG['log_k_standard_deviation'])
    amps=CONFIG['source_median_mg_L']*np.exp(CONFIG['source_log_standard_deviation']*rng.normal(size=3))
    truth=solve_basis(logk,times=CONFIG['times_days'],dt_days=dt_days,refinement=refinement)
    b=truth['basis'];head=truth['head']
    initial_truth=np.r_[amps@sample_cells(b[0],INITIAL_CONC,refinement),sample_cells(head,INITIAL_HEAD,refinement)]
    candidate_truth=amps@sample_cells(b[1],CANDIDATES,refinement)
    q=float(amps@sample_cells(b[2],[RECEPTOR],refinement)[:,0])
    initial=initial_truth+initial_sigma()*rng.normal(size=5)
    observed=candidate_truth+CONFIG['concentration_noise_sd_mg_L']*rng.normal(size=24)
    w=posterior(library,initial);before=posterior_summary(library,w)
    train=PredictiveExperiment(library,w,CONFIG['design_outer_draws'],seed+1000000)
    plans={'coverage':(select_coverage(CONFIG['batch_size']),[],0.)}
    for name,criterion in METHODS:plans[name]=select_greedy(train,criterion,CONFIG['batch_size'])
    true_event=bool(q>CONFIG['future_threshold_mg_L']);rows=[];traces={}
    raw=dict(truth_log_k=logk,truth_source_amplitudes=amps,truth_initial=initial_truth,
             truth_candidate_concentrations=candidate_truth,truth_q=np.array(q),
             initial_data=initial,actual_new_data_all_candidates=observed,initial_posterior=w)
    for name,(selected,trace,elapsed) in plans.items():
        residual=(library['candidate_predictions'][:,selected]-observed[selected])/CONFIG['concentration_noise_sd_mg_L']
        after_w=normalized(np.log(np.maximum(w,1e-300))-.5*np.sum(residual**2,axis=1))
        after=posterior_summary(library,after_w);action=after['action_strengthen_monitoring']
        fp=bool(not true_event and action);fn=bool(true_event and not action)
        row=dict(group=group,seed=seed,method=name,truth_refinement=refinement,truth_dt_days=dt_days,
                 true_q_mg_L=q,true_exceedance=true_event,selected_indices=selected,
                 selected_wells=[f'W{i+1:02d}' for i in selected],
                 before_q025_mg_L=before['future_q025_mg_L'],before_q975_mg_L=before['future_q975_mg_L'],
                 before_interval_coverage=bool(before['future_q025_mg_L']<=q<=before['future_q975_mg_L']),
                 before_effective_support=before['effective_support_size'],
                 after_mean_mg_L=after['future_mean_mg_L'],after_q025_mg_L=after['future_q025_mg_L'],
                 after_q975_mg_L=after['future_q975_mg_L'],after_exceedance_probability=after['exceedance_probability'],
                 after_interval_coverage=bool(after['future_q025_mg_L']<=q<=after['future_q975_mg_L']),
                 action_strengthen_monitoring=action,false_positive=fp,false_negative=fn,
                 realized_weighted_loss=float(CONFIG['false_positive_cost']*fp+CONFIG['false_negative_cost']*fn),
                 after_effective_support=after['effective_support_size'],selection_seconds=elapsed,
                 truth_pde_seconds=truth['elapsed_s'],truth_mass_balance_relative=truth['mass_balance_relative'])
        rows.append(row);raw[name+'_posterior_after']=after_w;traces[name]=trace
    return rows,raw,dict(seed=seed,group=group,before=before,selection_traces=traces,
                        total_seconds=time.perf_counter()-start,truth_log_k=logk.tolist(),truth_source_amplitudes_mg_L=amps.tolist())

def summarize(rows):
    summaries=[]
    for group in sorted({r['group'] for r in rows}):
        for method in ['coverage']+[m for m,c in METHODS]:
            part=[r for r in rows if r['group']==group and r['method']==method]
            truth=np.array([r['true_q_mg_L'] for r in part]);pred=np.array([r['after_mean_mg_L'] for r in part])
            loss=np.array([r['realized_weighted_loss'] for r in part]);ess=np.array([r['after_effective_support'] for r in part])
            summaries.append(dict(group=group,method=method,n_cases=len(part),true_exceedance_count=sum(r['true_exceedance'] for r in part),
                before_95_interval_coverage=float(np.mean([r['before_interval_coverage'] for r in part])),
                after_95_interval_coverage=float(np.mean([r['after_interval_coverage'] for r in part])),
                false_positive_count=sum(r['false_positive'] for r in part),false_negative_count=sum(r['false_negative'] for r in part),
                realized_loss_mean=float(loss.mean()),realized_loss_total=float(loss.sum()),
                realized_loss_standard_error=float(loss.std(ddof=1)/np.sqrt(len(loss))) if len(loss)>1 else 0.,
                prediction_rmse_mg_L=float(np.sqrt(np.mean((truth-pred)**2))),
                after_effective_support_median=float(np.median(ess)),after_effective_support_min=float(ess.min()),
                mean_selection_seconds=float(np.mean([r['selection_seconds'] for r in part]))))
    return dict(status='Exploratory continuous-prior and mesh/time mismatch stress tests; synthetic observations; not heldout main results',
                nominal_interval_level=.95,threshold_mg_L=CONFIG['future_threshold_mg_L'],false_negative_cost=CONFIG['false_negative_cost'],
                false_positive_cost=CONFIG['false_positive_cost'],groups=summaries)

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/stress')
    p.add_argument('--library',type=Path,default=ROOT/'results/library.npz');p.add_argument('--skip-enumeration',action='store_true')
    p.add_argument('--development-case',type=Path,default=ROOT/'results/case_1102.npz',
                   help='Saved initial posterior NPZ for development case 1102 used by the enumeration audit')
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    library=dict(np.load(args.library));assert str(library['configuration_sha256'])==hash_config(CONFIG)
    manifest=dict(status='Exploratory supplement; protocol and seeds fixed before these runs',config_sha256=hash_config(CONFIG),
                  source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['physics.py','inference_design.py','stress_tests.py']},
                  same_model_seeds=list(range(4001,4065)),mismatch_seeds=list(range(5001,5017)),
                  enumeration_development_case=str(args.development_case.resolve()),
                  same_model_truth=dict(refinement=4,dt_days=2.),mismatch_truth=dict(refinement=8,dt_days=.5),
                  platform=platform.platform(),hardware='CPU; launched with single-thread BLAS settings',
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(out/'manifest.json',manifest)
    if not args.skip_enumeration:enumeration(library,out,args.development_case)
    allrows=[]
    for group,seeds,refinement,dt in [('continuous_same_model',range(4001,4065),4,2.),('mesh_time_mismatch',range(5001,5017),8,.5)]:
        for seed in seeds:
            rows,raw,detail=continuous_case(library,seed,group,refinement,dt)
            np.savez_compressed(out/f'raw_case_{seed}.npz',**raw);write_json(out/f'case_{seed}.json',detail)
            allrows.extend(rows);write_json(out/'results.json',allrows);write_csv(out/'results.csv',allrows)
            write_json(out/'summary.json',summarize(allrows))
            print(json.dumps({'event':'stress_case_complete','group':group,'seed':seed,'seconds':detail['total_seconds'],
                              'true_q':rows[0]['true_q_mg_L'],'loss':{r['method']:r['realized_weighted_loss'] for r in rows}}),flush=True)
    print(json.dumps({'event':'stress_complete','cases':len(allrows)//4,'rows':len(allrows)}),flush=True)

if __name__=='__main__':main()

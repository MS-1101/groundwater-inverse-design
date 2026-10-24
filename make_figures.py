"""Rebuild eight scientific figures and a review-required well report from saved runs.

No new simulations, fitted data or generated illustrations are used here.
"""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from inference_design import CANDIDATES, INITIAL_CONC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
METHODS = ['coverage', 'parameter_information', 'prediction_variance', 'decision_loss']
LABELS = ['Coverage', 'Parameter\ninformation', 'Predictive\nvariance', 'Decision\nloss']
COLORS = ['#929ea8', '#377ca8', '#b88c34', '#178d82']


def read(path):
    return json.loads(Path(path).read_text())


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=260, bbox_inches='tight')
    plt.close(fig)


def bars(ax, values, title, ylabel, errors=None):
    ax.bar(range(4), values, color=COLORS, yerr=errors, capsize=3)
    ax.set_xticks(range(4), LABELS)
    ax.set_title(title, loc='left', weight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.titlesize':11, 'svg.fonttype':'none'})
    lib = dict(np.load(ROOT/'results/library.npz'))
    rows = read(ROOT/'results/heldout_results.json')
    case = read(ROOT/'results/case_3001.json')
    raw = dict(np.load(ROOT/'results/case_3001.npz'))
    stress = read(ROOT/'results/stress_summary.json')['groups']
    physics = read(ROOT/'physics_results/validation.json')
    fine = read(ROOT/'physics_results/fine_grid_check.json')
    toy = read(ROOT/'examples/math_examples.json')
    enum = read(ROOT/'results/stress_enum.json')
    r = int(lib['library_refinement'])
    plans = {x['method']: x['selected_indices'] for x in case['rows']}
    def field(w, ti):
        a = np.einsum('ka,kas->ks', w.reshape(9,125), lib['parameters'][:,2:].reshape(9,125,3))
        return np.einsum('ks,ksij->ij', a, lib['basis'][:,ti])
    def xy(cells):
        cells=np.asarray(cells)
        return (cells[:,1]+.5)*100, 1800-(cells[:,0]+.5)*100

    # 01: a fixed, reproducible example, not a selected best case.
    fig, axes = plt.subplots(1,2,figsize=(11,6),layout='constrained')
    for ax,w,title in zip(axes,[raw['initial_posterior'],raw['decision_loss_posterior_after']],
                          ['A  Day-365 predicted plume before sampling','B  Day-365 plume after one simulated batch']):
        im=ax.imshow(np.maximum(field(w,1),1e-4),extent=[0,1400,0,1800],origin='upper',
                     cmap='viridis',norm=LogNorm(vmin=1e-4,vmax=10))
        ax.scatter(*xy(CANDIDATES),s=24,facecolors='none',edgecolors='white',label='24 candidates')
        ax.scatter(*xy(INITIAL_CONC),s=40,marker='x',c='#ffca57',label='Initial concentration data')
        chosen=CANDIDATES[plans['decision_loss']]
        ax.scatter(*xy(chosen),s=75,facecolors='none',edgecolors='#f66b5d',linewidths=2,label='Selected batch')
        for ix,(x,y) in zip(plans['decision_loss'],zip(*xy(chosen))):
            ax.annotate(f'W{ix+1:02}',(x,y),xytext=(7,6),textcoords='offset points',color='white',fontsize=8)
        ax.scatter(650,750,marker='*',s=110,c='white',edgecolors='#182a35',label='Future receptor')
        ax.set(xlabel='x (m)',ylabel='y (m)',title=title)
    axes[0].legend(loc='upper right',fontsize=7,framealpha=.9)
    fig.colorbar(im,ax=axes,label='Posterior mean concentration (mg/L)',shrink=.75)
    fig.suptitle('Fixed example: case 3001 · finite-model candidate report · review required',fontsize=12)
    save(fig,'01_plume_and_candidate_plan')

    # Aggregate summaries: case means are the unit of between-case uncertainty.
    summary=[]
    for m in METHODS:
        rr=[x for x in rows if x['method']==m]
        summary.append({'method':m,'cases':len(rr),
                        'mean_expected_risk':float(np.mean([x['decision_risk'] for x in rr])),
                        'mean_case_rmse_mg_L':float(np.mean([x['predictive_rmse_mg_L'] for x in rr])),
                        'false_negatives':sum(x['false_negative_count'] for x in rr),
                        'false_positives':sum(x['false_positive_count'] for x in rr),
                        'selection_seconds':None if m=='coverage' else float(np.mean([x['selection_seconds'] for x in rr]))})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    means=[x['mean_expected_risk'] for x in summary]
    bars(axes[0],means,'A  Internal finite-model expected loss','Expected weighted loss')
    axes[0].set_yscale('log');axes[0].set_ylim(1e-7,.03)
    for i,v in enumerate(means):axes[0].text(i,v*1.35,f'{v:.2e}',ha='center',fontsize=9)
    bars(axes[1],[x['mean_case_rmse_mg_L'] for x in summary],
         'B  Prediction accuracy has a different ordering','Mean case RMSE (mg/L)')
    axes[1].set_ylim(0,.145)
    fig.suptitle('12 held-out cases × 4,096 shared conditional predictive draws per method',fontsize=12)
    save(fig,'02_internal_risk_and_prediction')

    # 03: a causal invariance check and a later genuinely informative update.
    a3=lib['parameters'][:,4];nodes=np.unique(a3)
    weights=[lib['prior'],raw['initial_posterior'],raw['decision_loss_posterior_after']]
    probs=[[float(w[a3==v].sum()) for v in nodes] for w in weights]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for p,lab,col,marker in zip(probs,['Prior','After day-180 data','After day-365 data'],
                               ['#929ea8','#377ca8','#178d82'],['o','x','s']):
        axes[0].plot(nodes,p,label=lab,c=col,marker=marker)
    axes[0].set(xlabel='Third release amplitude (mg/L)',ylabel='Marginal probability',title='A  Case 3001: third pulse starts at day 240')
    axes[0].legend(fontsize=8)
    errors=[]
    for seed in [1101,1102,1103]+list(range(3001,3013)):
        stage='development' if seed<2000 else 'heldout'
        w=np.load(ROOT/f'results/case_{seed}.npz')['initial_posterior']
        errors.append(max(abs(w[a3==v].sum()-lib['prior'][a3==v].sum()) for v in nodes))
    axes[1].plot(range(1,16),np.asarray(errors)*1e16,'o-',c='#377ca8')
    axes[1].set(xlabel='Case index (3 development + 12 held-out)',ylabel='Maximum marginal change (× 10⁻¹⁶)',
                title='B  Invariance holds to floating-point precision')
    axes[1].grid(alpha=.2)
    save(fig,'03_source_timing_independence')

    # 04 and 06: unfavorable tests stay visible, with case-based standard errors.
    for group,n,name in [('continuous_same_model',64,'04_continuous_truth_calibration'),
                         ('mesh_time_mismatch',16,'06_mesh_time_mismatch')]:
        ss=[next(x for x in stress if x['group']==group and x['method']==m) for m in METHODS]
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        bars(axes[0],[x['realized_loss_mean'] for x in ss],'A  Realized decision loss','Weighted loss per case',
             [x['realized_loss_standard_error'] for x in ss])
        axes[0].set_ylim(bottom=0)
        bars(axes[1],[100*x['after_95_interval_coverage'] for x in ss],
             'B  Nominal 95% interval coverage','Cases covered (%)')
        axes[1].axhline(95,c='#ac3847',ls='--',label='Nominal 95%')
        axes[1].axhline(100*ss[0]['before_95_interval_coverage'],c='#475d6c',ls=':',label='Before new samples')
        axes[1].set_ylim(0,105);axes[1].legend(fontsize=8,loc='upper right')
        for i,x in enumerate(ss):
            axes[1].text(i,100*x['after_95_interval_coverage']+2,
                         f"{round(n*x['after_95_interval_coverage'])}/{n}",ha='center',fontsize=9)
        subtitle='Continuous-prior truths; same grid and time step' if n==64 else 'Continuous truths on 12.5 m / 0.5 d; inversion on 25 m / 2 d'
        fig.suptitle(f'{n} exploratory synthetic cases · {subtitle}',fontsize=12)
        save(fig,name)

    # 05: mesh sensitivity and the separately controlled time-step test.
    grid=physics['grid_refinement'][:2]+fine
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    axes[0].plot([x['dx_m'] for x in grid],[x['receptor_545_mg_L'] for x in grid],'o-',c='#ac3847')
    axes[0].set_xscale('log',base=2);axes[0].invert_xaxis()
    axes[0].set_xticks([100,50,25,12.5,6.25],['100','50','25','12.5','6.25'])
    axes[0].axvline(25,c='#64717d',ls=':',label='Main grid')
    axes[0].set(xlabel='Cell width (m), fixed Δt = 0.5 d',ylabel='Day-545 receptor concentration (mg/L)',
                title='A  Receptor tail remains resolution-sensitive')
    axes[0].legend(fontsize=8)
    t=physics['time_refinement']
    axes[1].plot([x['dt_days'] for x in t],[100*x['relative_l2_vs_quarter_day'] for x in t],'o-',c='#377ca8')
    axes[1].set_xscale('log',base=2);axes[1].invert_xaxis()
    axes[1].set_xticks([8,4,2,1,.5],['8','4','2','1','0.5'])
    axes[1].set(xlabel='Time step (d), fixed 100 m grid',ylabel='Relative field L2 error vs Δt = 0.25 d (%)',
                title='B  Separate temporal refinement experiment')
    fig.suptitle('Fixed numerical probe: log K = (0.2, −0.2), amplitudes = (45, 75, 30) mg/L',fontsize=12)
    save(fig,'05_numerical_resolution')

    # 07: teaching calculations, explicitly not physical performance results.
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    g=toy['gaussian']
    axes[0].plot(g['posterior_variances'],g['expected_risks'],'o-',c='#178d82')
    axes[0].set(xlabel='Conditional posterior variance',ylabel='Preposterior optimal Bayes risk',
                title='A  Same scalar Gaussian target: compatible ordering')
    ng=toy['non_gaussian'];v=[ng[k]['expected_variance'] for k in ['observe_x1','observe_x2']]
    risk=[ng[k]['expected_bayes_risk'] for k in ['observe_x1','observe_x2']]
    axes[1].scatter(v,risk,s=70,c=['#b88c34','#377ca8'])
    for x,y,label in zip(v,risk,['Observe X1','Observe X2']):axes[1].annotate(label,(x,y),xytext=(7,9),textcoords='offset points')
    axes[1].set(xlabel='Expected posterior variance',ylabel='Expected Bayes risk',xlim=(-1,28),ylim=(-.025,.31),
                title='B  Discrete non-Gaussian counterexample')
    fig.suptitle('Executed mathematical teaching examples · distinct distributions and thresholds',fontsize=12)
    save(fig,'07_gaussian_and_discrete_teaching')

    # 08: exhaustive objective audit, and measured selection timings only.
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    scores=sorted(x['decision_risk'] for x in enum['all_training_scores'])
    axes[0].plot(range(1,221),scores,c='#377ca8')
    axes[0].scatter(1,scores[0],s=65,c='#178d82',zorder=3)
    axes[0].set_yscale('log');axes[0].set(xlabel='Rank among all 220 batches',ylabel='Training Monte Carlo expected loss',
                                          title='A  12 choose 3: greedy matches best sampled objective')
    axes[0].text(.05,.92,'Same set: W03, W11, W21',transform=axes[0].transAxes,fontsize=9)
    vals=[x['selection_seconds'] for x in summary[1:]]
    axes[1].bar(range(3),vals,color=COLORS[1:])
    axes[1].set_xticks(range(3),LABELS[1:]);axes[1].set_ylim(0,.82)
    axes[1].set(ylabel='Mean selection wall time (s)',title='B  Main selection cost, existing response library')
    for i,v in enumerate(vals):axes[1].text(i,v+.015,f'{v:.3f} s',ha='center')
    axes[1].text(.5,.92,'Coverage timing was not instrumented',ha='center',transform=axes[1].transAxes,fontsize=8)
    save(fig,'08_search_audit_and_cost')

    # Machine-readable aggregate and a concrete engineering handoff object.
    paired=[]
    for seed in range(3001,3013):
        z=np.load(ROOT/f'results/case_{seed}.npz')
        d=z['decision_loss_risk_by_draw']-z['prediction_variance_risk_by_draw']
        paired.append(d)
    conditional_se=float(np.sqrt(sum(np.var(d,ddof=1)/len(d) for d in paired))/12)
    report={'status':'Synthetic benchmark research; not field validated', 'main':summary,
            'paired_decision_minus_variance_risk':float(np.mean(paired)),
            'paired_conditional_mcse':conditional_se,
            'between_case_se_of_paired_means':float(np.std([np.mean(d) for d in paired],ddof=1)/np.sqrt(12)),
            'stress':stress,'configuration_sha256':str(lib['configuration_sha256']),
            'figure_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'results/figure_data_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    with (ROOT/'results/sampling_plan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['case_seed','construction_order','well','x_m','y_m','sampling_day','initial_predictive_mean_mg_L',
                                              'initial_predictive_sd_mg_L','objective','status'])
        w=raw['initial_posterior']
        for rank,idx in enumerate(plans['decision_loss'],1):
            c=lib['candidate_predictions'][:,idx];mean=float(w@c);sd=float(np.sqrt(w@((c-mean)**2)))
            x,y=xy(CANDIDATES[[idx]])
            writer.writerow([3001,rank,f'W{idx+1:02}',float(x[0]),float(y[0]),365,mean,sd,
                             'Expected batch decision loss; all four wells sampled together','Research candidate; continuous-truth calibration requires reassessment'])
    with (ROOT/'results/candidate_wells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['well','base_grid_row_zero_based','base_grid_column_zero_based','x_m','y_m','sampling_day','position_type'])
        for idx,cell in enumerate(CANDIDATES):
            x,y=xy([cell])
            writer.writerow([f'W{idx+1:02}',int(cell[0]),int(cell[1]),float(x[0]),float(y[0]),365,'Virtual 100 m monitoring footprint'])
    with (ROOT/'results/plan_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['case_seed','method','construction_order','well','x_m','y_m','independent_finite_model_expected_loss','status'])
        for row in case['rows']:
            for rank,idx in enumerate(row['selected_indices'],1):
                x,y=xy(CANDIDATES[[idx]])
                writer.writerow([3001,row['method'],rank,f'W{idx+1:02}',float(x[0]),float(y[0]),row['decision_risk'],
                                 'Research candidate; posterior calibration and numerical resolution require reassessment'])
    print(f'Created 8 PNG figures in {OUT}')


if __name__=='__main__':
    main()

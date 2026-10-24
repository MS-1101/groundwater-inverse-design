"""Analytic solution, conservation, positivity, superposition and resolution audit."""
from pathlib import Path
import argparse, json, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from physics import Config, solve_flow, solve_basis, transport_operator, coarsen, sample_cells

ROOT=Path(__file__).resolve().parent

def diffusion_exact(refinement):
    """Closed rectangular aquifer, zero advection, decaying Neumann cosine mode."""
    r=refinement; config=Config(refinement=r,alpha_longitudinal_m=0.,alpha_transverse_m=0.,diffusion_m2_day=100.)
    f=solve_flow(config=config)
    f['flow'][:]=0.; f['well_q'][:]=0.; f['external_q'][:]=0.
    op=transport_operator(f,config); nr,nc=f['head'].shape; dx=100./r
    x=(np.arange(nc)+.5)*dx; y=(np.arange(nr)+.5)*dx
    mode=np.cos(np.pi*y[:,None]/1800)*np.cos(np.pi*x[None,:]/1400)
    initial=1.+.5*mode; c=initial.ravel().copy()
    dt=4./r**2; duration=400.; vol=op['volume']
    A=sparse.diags(vol/dt)+op['L']; lu=splu(A.tocsc())
    for _ in range(round(duration/dt)): c=lu.solve(vol/dt*c)
    c=c.reshape(nr,nc)
    exact=1.+.5*mode*np.exp(-100.*np.pi**2*(1./1400**2+1./1800**2)*duration)
    return dict(refinement=r,dx_m=dx,dt_days=dt,rmse_mg_L=float(np.sqrt(np.mean((c-exact)**2))),
                mass_relative_error=float(abs((c-initial).sum())/initial.sum()),minimum=float(c.min()))

def linear_head():
    # Separate exact fixture assembled from the same face-conductance stencil.
    config=Config(); f=solve_flow(config=config); nr,nc=f['head'].shape; n=nr*nc
    p,z=f['p'],f['z']; g=np.full(len(p),config.k_background_m_day*config.thickness_m)
    A=sparse.coo_matrix((np.r_[g,g,-g,-g],(np.r_[p,z,p,z],np.r_[p,z,z,p])),shape=(n,n)).tocsr()
    exact=np.broadcast_to(np.linspace(250.,25.,nr)[:,None],(nr,nc)).copy().ravel()
    known=np.r_[np.arange(nc),np.arange(n-nc,n)]; unknown=np.setdiff1d(np.arange(n),known)
    from scipy.sparse.linalg import spsolve
    h=exact.copy(); h[unknown]=spsolve(A[unknown][:,unknown],-A[unknown][:,known]@h[known])
    return dict(max_head_error_m=float(abs(h-exact).max()),no_flow_side_boundary=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'physics_results')
    out=ap.parse_args().out; out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); times=(180.,365.,545.)
    # This is a fixed numerical test case, not an inferred or observed field case.
    log_k=(.2,-.2); amplitudes=np.array([45.,75.,30.])
    nominal=solve_basis(log_k,times=times,dt_days=2.)
    held_cases=[]
    for a in (-.5,0.,.5):
        for b in (-.5,0.,.5):
            r=solve_basis((a,b),times=times,dt_days=2.)
            held_cases.append(dict(log_k=[a,b],mass_balance_relative=r['mass_balance_relative'],
                                   flow_residual_m3_day=r['flow_residual_m3_day'],minimum_concentration=r['minimum_concentration']))
    diffusion=[diffusion_exact(r) for r in (1,2,4)]
    for i in (1,2): diffusion[i]['observed_order']=float(np.log(diffusion[i-1]['rmse_mg_L']/diffusion[i]['rmse_mg_L'])/np.log(2))
    # Time and mesh differences are numerical sensitivity evidence, not exact errors.
    temporal=[]; time_fields={}
    ref=solve_basis(log_k,times=times,dt_days=.25)['basis']
    for dt in (8.,4.,2.,1.,.5):
        r=solve_basis(log_k,times=times,dt_days=dt); field=np.einsum('tsij,s->tij',r['basis'],amplitudes)
        diff=np.einsum('tsij,s->tij',r['basis']-ref,amplitudes)
        temporal.append(dict(dt_days=dt,relative_l2_vs_quarter_day=float(np.linalg.norm(diff)/np.linalg.norm(np.einsum('tsij,s->tij',ref,amplitudes))),
                        receptor_545_mg_L=float(field[-1,10,6]),mass_balance_relative=r['mass_balance_relative']))
        time_fields[f'dt_{str(dt).replace(".","p")}']=field
    grids=[]; grid_fields={}; basis_coarse={}
    for refinement in (1,2,4):
        r=solve_basis(log_k,times=times,dt_days=.5,refinement=refinement)
        basis_coarse[refinement]=coarsen(r['basis'],refinement)
        field=np.einsum('tsij,s->tij',basis_coarse[refinement],amplitudes)
        grid_fields[f'refinement_{refinement}']=field
        grids.append(dict(refinement=refinement,dx_m=100./refinement,dt_days=.5,
                          receptor_545_mg_L=float(field[-1,10,6]),mass_balance_relative=r['mass_balance_relative'],elapsed_s=r['elapsed_s']))
    for i,r in enumerate((1,2,4)):
        d=np.einsum('tsij,s->tij',basis_coarse[r]-basis_coarse[4],amplitudes)
        grids[i]['relative_l2_vs_25m']=float(np.linalg.norm(d)/np.linalg.norm(grid_fields['refinement_4']))
    # All amplitudes equal: three 120-day pulses equal a single 360-day source.
    union=solve_basis(log_k,times=times,dt_days=2.,pulse_edges=(0.,360.))['basis'][:,0]
    superposition=float(abs(nominal['basis'].sum(axis=1)-union).max())
    L=transport_operator(nominal['flow'])['L']; off=L-sparse.diags(L.diagonal())
    report=dict(scope='Synthetic benchmark-derived numerical validation; CPU execution, no field data.',
                test_log_k=list(log_k),test_amplitudes_mg_L=amplitudes.tolist(),times_days=list(times),
                nominal=dict(mass_balance_relative=nominal['mass_balance_relative'],
                             flow_residual_m3_day=nominal['flow_residual_m3_day'],
                             minimum_concentration=nominal['minimum_concentration']),
                linear_head=linear_head(),analytic_diffusion=diffusion,
                superposition_max_absolute_mg_L=superposition,
                maximum_positive_transport_offdiagonal=float(max(0.,off.max())),
                parameter_corner_checks=held_cases,time_refinement=temporal,grid_refinement=grids,
                refinement_definition='100 m source and observation cell footprints retained; top/bottom 100 m strips keep Dirichlet heads; concentration observations use footprint averages.',
                elapsed_total_s=time.perf_counter()-start)
    assert report['linear_head']['max_head_error_m']<1e-8
    assert all(x['observed_order']>1.9 for x in diffusion[1:])
    assert superposition<1e-12
    assert max(x['mass_balance_relative'] for x in held_cases)<1e-10
    assert min(x['minimum_concentration'] for x in held_cases)>=-1e-12
    assert report['maximum_positive_transport_offdiagonal']==0
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    (out/'nominal_mass_budget.json').write_text(json.dumps(nominal['budgets'],indent=2))
    np.savez_compressed(out/'resolution_fields.npz',**time_fields,**grid_fields,times_days=times)
    np.savez_compressed(out/'nominal_basis.npz',head=nominal['head'],basis=nominal['basis'],times_days=times)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

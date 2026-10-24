"""Conservative finite volumes for a benchmark-derived groundwater scenario.

All computations use metres and days. Concentration is numerically mg/L;
mass diagnostics multiply concentration * m3 by 1000 L/m3 to give mg.
This is steady Darcy + first-order upwind + diagonal hydrodynamic dispersion,
not the TVD/full-dispersion solver in the original MT3DMS problem 9 example.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

@dataclass(frozen=True)
class Config:
    refinement: int = 1
    thickness_m: float = 10.
    porosity: float = .3
    k_background_m_day: float = 1.474e-4 * 86400
    k_lens_m_day: float = 1.474e-7 * 86400
    injection_m3_day: float = .001 * 86400
    extraction_m3_day: float = .0189 * 86400
    alpha_longitudinal_m: float = 20.
    alpha_transverse_m: float = 4.
    diffusion_m2_day: float = 0.

def _grid(config, log_k):
    r = config.refinement
    if r < 1 or int(r) != r: raise ValueError('refinement must be a positive integer')
    nr, nc, dx = 18*r, 14*r, 100./r
    k = np.full((nr,nc), config.k_background_m_day*np.exp(log_k[0]))
    k[5*r:8*r, r:8*r] = config.k_lens_m_day*np.exp(log_k[1])
    q = np.zeros((nr,nc))
    q[3*r:4*r,6*r:7*r] = config.injection_m3_day/r**2
    q[10*r:11*r,6*r:7*r] = -config.extraction_m3_day/r**2
    fixed = np.zeros((nr,nc),bool); fixed[:r]=True; fixed[-r:]=True
    hv = np.zeros((nr,nc)); hv[:r]=250.
    hv[-r:] = 20.+(((np.arange(nc)+.5)*dx-50.)/100)*2.5
    ids=np.arange(nr*nc).reshape(nr,nc)
    p=np.r_[ids[:,:-1].ravel(),ids[:-1,:].ravel()]
    z=np.r_[ids[:,1:].ravel(),ids[1:,:].ravel()]
    horizontal=np.r_[np.ones(nr*(nc-1),bool),np.zeros((nr-1)*nc,bool)]
    kval=k.ravel(); g=2*kval[p]*kval[z]/(kval[p]+kval[z])*config.thickness_m
    return nr,nc,dx,k,q,fixed,hv,p,z,horizontal,g

def solve_flow(log_k=(0.,0.), config=None):
    """Return heads, oriented face flows, external CHD exchange and water residual."""
    config = Config() if config is None else config
    nr,nc,dx,k,q,fixed,hv,p,z,horiz,g=_grid(config,log_k)
    n=nr*nc
    A=sparse.coo_matrix((np.r_[g,g,-g,-g],(np.r_[p,z,p,z],np.r_[p,z,z,p])),shape=(n,n)).tocsr()
    unknown=np.flatnonzero(~fixed.ravel()); known=np.flatnonzero(fixed.ravel())
    h=hv.ravel().copy()
    h[unknown]=spsolve(A[unknown][:,unknown],q.ravel()[unknown]-A[unknown][:,known]@h[known])
    flow=g*(h[p]-h[z]); outgoing=A@h
    external=np.zeros(n); external[known]=outgoing[known]-q.ravel()[known]
    residual=outgoing-q.ravel()-external
    return dict(head=h.reshape(nr,nc),k=k,well_q=q,external_q=external.reshape(nr,nc),
                p=p,z=z,horizontal=horiz,flow=flow,conductance=g,dx=dx,config=asdict(config),
                flow_residual_m3_day=float(np.max(np.abs(residual))),
                global_flow_residual_m3_day=float(abs((q.ravel()+external).sum())))

def transport_operator(flow, config=None):
    """M dC/dt + L C = Qinj u; zero exterior inflow concentration, free outflow.

    L has non-positive off-diagonals. Internal advective/dispersive fluxes cancel
    pairwise; only extraction and external water outflow remove contaminant.
    """
    config=Config(**flow['config']) if config is None else config
    h=flow['head']; nr,nc=h.shape; n=h.size
    p,z,f=flow['p'],flow['z'],flow['flow']
    qx=np.zeros((nr,nc+1)); qy=np.zeros((nr+1,nc))
    cut=nr*(nc-1); qx[:,1:-1]=f[:cut].reshape(nr,nc-1); qy[1:-1]=f[cut:].reshape(nr-1,nc)
    # Zero side boundary flow. For fixed-head strips, internal averages are used.
    vx=.5*(qx[:,:-1]+qx[:,1:])/(config.thickness_m*flow['dx']*config.porosity)
    vy=.5*(qy[:-1]+qy[1:])/(config.thickness_m*flow['dx']*config.porosity)
    speed=np.hypot(vx,vy); denom=np.maximum(speed,1e-30)
    dxx=config.diffusion_m2_day+(config.alpha_longitudinal_m*vx**2+config.alpha_transverse_m*vy**2)/denom
    dyy=config.diffusion_m2_day+(config.alpha_transverse_m*vx**2+config.alpha_longitudinal_m*vy**2)/denom
    dc=np.where(flow['horizontal'],.5*(dxx.ravel()[p]+dxx.ravel()[z]),.5*(dyy.ravel()[p]+dyy.ravel()[z]))
    disp=dc*config.porosity*config.thickness_m
    pos=np.maximum(f,0); neg=np.maximum(-f,0)
    sink=np.maximum(-flow['well_q'].ravel(),0)+np.maximum(-flow['external_q'].ravel(),0)
    rows=np.r_[p,z,z,p,p,z,p,z,np.arange(n)]
    cols=np.r_[p,p,z,z,p,z,z,p,np.arange(n)]
    vals=np.r_[pos,-pos,neg,-neg,disp,disp,-disp,-disp,sink]
    L=sparse.coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsc()
    volume=np.full(n,config.porosity*config.thickness_m*flow['dx']**2)
    source=np.maximum(flow['well_q'].ravel(),0)
    return dict(L=L,volume=volume,source=source,sink=sink,vx=vx,vy=vy,dxx=dxx,dyy=dyy)

def solve_basis(log_k=(0.,0.), times=(180.,365.,730.), dt_days=2., refinement=1,
                config=None, pulse_edges=(0.,120.,240.,360.)):
    """Three unit injection-concentration responses: shape (time,pulse,row,col).

    log_k are natural log multipliers of background/lens K, not log absolute K.
    Backward Euler steps are split exactly at requested observations and pulse
    transitions, so no discontinuity is crossed by an unsplit source step.
    """
    started=time.perf_counter()
    config=Config(refinement=refinement) if config is None else config
    times=np.array(times,float); edges=np.array(pulse_edges,float)
    if np.any(np.diff(times)<=0) or times[0]<=0 or dt_days<=0: raise ValueError('invalid time grid')
    if len(edges)<2 or edges[0]!=0 or np.any(np.diff(edges)<=0): raise ValueError('invalid pulses')
    flow=solve_flow(log_k,config); op=transport_operator(flow,config)
    n=flow['head'].size; ns=len(edges)-1
    c=np.zeros((n,ns)); volume=op['volume']; L=op['L']; factors={}
    cumulative_in=np.zeros(ns); cumulative_out=np.zeros(ns); snapshots=[]; budgets=[]
    t=0.; maximum=times[-1]; events=sorted(set(times.tolist()+edges[(edges>0)&(edges<maximum)].tolist()+[maximum]))
    for end in events:
        while t < end-1e-9:
            dt=min(dt_days,end-t); key=round(float(dt),10)
            if key not in factors: factors[key]=splu(sparse.diags(volume/dt)+L)
            mid=t+.5*dt; active=np.array([(a<=mid<b) for a,b in zip(edges[:-1],edges[1:])],float)
            rhs=volume[:,None]/dt*c+op['source'][:,None]*active[None,:]
            c=factors[key].solve(rhs)
            cumulative_in += dt*op['source'].sum()*active
            cumulative_out += dt*(op['sink']@c)
            t+=dt
        if np.any(np.isclose(times,end,rtol=0,atol=1e-8)):
            snapshots.append(c.T.reshape(ns,*flow['head'].shape).copy())
            stored=volume@c; residual=stored+cumulative_out-cumulative_in
            budgets.append(dict(time_days=float(end),injected_mg=(cumulative_in*1000).tolist(),
                                extracted_and_boundary_mg=(cumulative_out*1000).tolist(),stored_mg=(stored*1000).tolist(),
                                residual_mg=(residual*1000).tolist(),relative_error=(abs(residual)/np.maximum(cumulative_in,1)).tolist()))
    result=dict(head=flow['head'],basis=np.array(snapshots),times_days=times,log_k=np.array(log_k),
                flow_residual_m3_day=flow['flow_residual_m3_day'],
                global_flow_residual_m3_day=flow['global_flow_residual_m3_day'],
                mass_balance_relative=max(max(b['relative_error']) for b in budgets),budgets=budgets,
                minimum_concentration=float(c.min()),config=asdict(config),dt_days=dt_days,
                elapsed_s=time.perf_counter()-started,flow=flow)
    return result

def sample_cells(field,cells,refinement=1):
    """Average each original 100 m cell footprint; preserve all leading axes."""
    a=np.asarray(field); r=refinement
    return np.stack([a[...,i*r:(i+1)*r,j*r:(j+1)*r].mean(axis=(-2,-1)) for i,j in cells],axis=-1)

def coarsen(field,refinement):
    a=np.asarray(field); r=refinement
    return a.reshape(*a.shape[:-2],18,r,14,r).mean(axis=(-3,-1))

if __name__=='__main__':
    import json
    x=solve_basis()
    print(json.dumps({k:x[k] for k in ('mass_balance_relative','flow_residual_m3_day','minimum_concentration','elapsed_s')},indent=2))

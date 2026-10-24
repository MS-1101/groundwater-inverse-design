"""Run the public P09 MF6 configuration, then a matched-equation FV audit.

The official source URL and its SHA256 are recorded below. This script
reproduces the MODFLOW 6 configuration, not the separate MT3DMS executable.
"""
from pathlib import Path
import argparse, json, hashlib, platform, subprocess, time
import numpy as np
import flopy
from physics import Config, solve_basis, solve_flow

ROOT=Path(__file__).resolve().parent
SOURCE='https://raw.githubusercontent.com/MODFLOW-ORG/modflow6-examples/ec5ac6134d3d25684fbdd5f204864258610f2330/scripts/ex-gwt-mt3dms-p09.py'

def run_case(name, exe, matched=False, out=ROOT/'physics_results'):
    ws=out/name; ws.mkdir(parents=True,exist_ok=True)
    sim=flopy.mf6.MFSimulation(sim_name=name,sim_ws=ws,exe_name=str(exe))
    flopy.mf6.ModflowTdis(sim,nper=2,perioddata=[(365.*86400,365,1.),(365.*86400,365,1.)],time_units='SECONDS')
    gwf=flopy.mf6.ModflowGwf(sim,modelname='flow',save_flows=True)
    ims=flopy.mf6.ModflowIms(sim,filename='flow.ims',outer_dvclose=1e-10,inner_dvclose=1e-10,
                            outer_maximum=100,inner_maximum=500,rcloserecord=1e-10,linear_acceleration='CG')
    sim.register_ims_package(ims,[gwf.name])
    dis=dict(nlay=1,nrow=18,ncol=14,delr=100.,delc=100.,top=0.,botm=-10.)
    flopy.mf6.ModflowGwfdis(gwf,length_units='METERS',**dis)
    k=np.full((1,18,14),1.474e-4); k[:,5:8,1:8]=1.474e-7
    initial=np.zeros((1,18,14)); initial[:,0]=250.; initial[:,-1]=20.+2.5*np.arange(14)
    if matched: initial=solve_flow()['head'][None]
    flopy.mf6.ModflowGwfic(gwf,strt=initial)
    flopy.mf6.ModflowGwfnpf(gwf,icelltype=0,k=k,k33=k,save_specific_discharge=True)
    sto_args=dict(ss=1e-5)
    if matched: sto_args['steady_state']={0:True}
    flopy.mf6.ModflowGwfsto(gwf,**sto_args)
    chd=[]
    for j in range(14):
        chd += [((0,0,j),250.,0.),((0,17,j),20.+2.5*j,0.)]
    flopy.mf6.ModflowGwfchd(gwf,stress_period_data={0:chd},auxiliary='CONCENTRATION',pname='CHD-1')
    flopy.mf6.ModflowGwfwel(gwf,auxiliary='CONCENTRATION',pname='WEL-1',stress_period_data={
        0:[((0,3,6),.001,57.87),((0,10,6),-.0189,0.)],
        1:[((0,3,6),.001,0.),((0,10,6),-.0189,0.)]})
    flopy.mf6.ModflowGwfoc(gwf,head_filerecord='flow.hds',budget_filerecord='flow.cbc',
                          saverecord=[('HEAD','LAST'),('BUDGET','LAST')])
    gwt=flopy.mf6.ModflowGwt(sim,modelname='transport',save_flows=True)
    imst=flopy.mf6.ModflowIms(sim,filename='transport.ims',outer_dvclose=1e-10,inner_dvclose=1e-10,
                             outer_maximum=100,inner_maximum=500,rcloserecord=1e-10,linear_acceleration='BICGSTAB')
    sim.register_ims_package(imst,[gwt.name])
    flopy.mf6.ModflowGwtdis(gwt,**dis)
    flopy.mf6.ModflowGwtic(gwt,strt=0.)
    flopy.mf6.ModflowGwtadv(gwt,scheme='UPSTREAM' if matched else 'TVD')
    if not matched: flopy.mf6.ModflowGwtdsp(gwt,alh=20.,ath1=4.)
    flopy.mf6.ModflowGwtmst(gwt,porosity=.3)
    flopy.mf6.ModflowGwtssm(gwt,sources=[('WEL-1','AUX','CONCENTRATION'),('CHD-1','AUX','CONCENTRATION')])
    flopy.mf6.ModflowGwtoc(gwt,concentration_filerecord='transport.ucn',budget_filerecord='transport.cbc',
                          saverecord=[('CONCENTRATION','LAST'),('BUDGET','LAST')])
    flopy.mf6.ModflowGwfgwt(sim,exgtype='GWF6-GWT6',exgmnamea='flow',exgmnameb='transport')
    sim.write_simulation(silent=True)
    t=time.perf_counter(); success,log=sim.run_simulation(silent=True,report=True); elapsed=time.perf_counter()-t
    (ws/'run.log').write_text('\n'.join(log))
    if not success: raise RuntimeError(f'MODFLOW failed: see {ws}/run.log')
    heads=gwf.output.head().get_alldata()[:,0]
    conc=gwt.output.concentration().get_alldata()[:,0]
    np.savez_compressed(ws/'fields.npz',head=heads,concentration=conc,times_days=[365,730])
    return dict(head=heads,concentration=conc,elapsed_s=elapsed,workspace=str(ws))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--exe',required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'physics_results');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    exe=Path(args.exe).resolve()
    original=run_case('mf6_original_p09',exe,out=args.out)
    matched=run_case('mf6_matched_upwind_zero_dispersion',exe,True,out=args.out)
    ours=solve_basis(times=(365.,730.),dt_days=1.,config=Config(alpha_longitudinal_m=0.,alpha_transverse_m=0.),pulse_edges=(0.,365.,730.))
    c=ours['basis'][:,0]*57.87; h=ours['head']
    c_diff=c-matched['concentration']; h_diff=h-matched['head'][-1]
    metrics=dict(
        executable_version=subprocess.check_output([str(exe),'-v'],text=True).strip(),
        executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
        official_download_url='https://github.com/MODFLOW-ORG/modflow6/releases/download/6.8.0/mf6.8.0_macarm.zip',
        source_url=SOURCE,source_sha256='be244863e2b88a3cd1d010744d7cff445a37445c8cc3c4ff5cbac17694b99bb1',
        platform=platform.platform(),flopy_version=flopy.__version__,
        original_configuration=dict(grid=[18,14],dt_days=1.,duration_days=730.,advection='TVD',
                                    dispersion='MODFLOW full tensor, alh=20 m, ath1=4 m',
                                    source_mg_L=57.87,source_interval_days=[0,365],
                                    differences_from_example='IMS closure tightened from 1e-6 to 1e-10 and inner iteration limit raised from 300 to 500; output file names changed; MT3DMS executable was not run'),
        original=dict(elapsed_s=original['elapsed_s'],concentration_max_mg_L=original['concentration'].max(axis=(1,2)).tolist(),
                      concentration_min_mg_L=original['concentration'].min(axis=(1,2)).tolist()),
        matched_equation_comparison=dict(equations='steady flow, upwind, no dispersion, dt=1 day; FV independent implementation',
                                        concentration_rmse_mg_L=float(np.sqrt(np.mean(c_diff**2))),
                                        concentration_max_abs_mg_L=float(abs(c_diff).max()),
                                        head_max_abs_m=float(abs(h_diff).max()),
                                        concentration_relative_l2=float(np.linalg.norm(c_diff)/np.linalg.norm(matched['concentration'])),
                                        elapsed_mf6_s=matched['elapsed_s'],elapsed_fv_s=ours['elapsed_s']))
    assert metrics['matched_equation_comparison']['concentration_relative_l2']<1e-6
    assert metrics['matched_equation_comparison']['head_max_abs_m']<1e-6
    np.savez_compressed(args.out/'matched_comparison.npz',fv_head=h,mf6_head=matched['head'][-1],
                        fv_concentration=c,mf6_concentration=matched['concentration'],original_concentration=original['concentration'],times_days=[365,730])
    (args.out/'mf6_audit.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()

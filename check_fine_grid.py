"""Repeat the fine-grid numerical sensitivity audit (roughly 20 s on this host)."""
from pathlib import Path
import argparse, json
import numpy as np
from physics import solve_basis,coarsen

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'physics_results')
    p=ap.parse_args().out; p.mkdir(parents=True,exist_ok=True); rows=[]; fields={}
    for r in (4,8,16):
        x=solve_basis((.2,-.2),times=(180.,365.,545.),dt_days=.5,refinement=r)
        f=np.einsum('tsij,s->tij',coarsen(x['basis'],r),[45.,75.,30.])
        fields[f'refinement_{r}']=f
        row=dict(refinement=r,dx_m=100/r,dt_days=.5,receptor_545_mg_L=float(f[-1,10,6]),
                 elapsed_s=x['elapsed_s'],mass_balance_relative=x['mass_balance_relative'])
        rows.append(row); print(json.dumps(row),flush=True)
    for row in rows:
        row['relative_l2_vs_6p25m']=float(np.linalg.norm(fields[f"refinement_{row['refinement']}"]-fields['refinement_16'])/np.linalg.norm(fields['refinement_16']))
    (p/'fine_grid_check.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(p/'fine_grid_check.npz',**fields)

if __name__=='__main__': main()

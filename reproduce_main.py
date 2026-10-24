"""Reproduce the frozen main study in a fresh directory; optional stress tests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results/reproduction')
    p.add_argument('--stress',action='store_true')
    args=p.parse_args();out=args.out.resolve()
    if out.exists():raise SystemExit('Choose a new directory; existing evidence is not overwritten.')
    out.mkdir(parents=True)
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[key]='1'
    def run(script,*values):
        command=[sys.executable,str(ROOT/script),*map(str,values)]
        print('RUN',command,flush=True)
        with (out/(script.removesuffix('.py')+'.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for group in ['development','heldout']:
        run('inference_design.py','--cases',group,'--out',out/group,'--library',out/'library.npz','--freeze',out/'protocol.json')
    run('check_inference.py','--results',out)
    if args.stress:
        run('stress_tests.py','--library',out/'library.npz','--out',out/'stress',
            '--development-case',out/'development/case_1102.npz')
    print('Complete:',out)


if __name__=='__main__':main()

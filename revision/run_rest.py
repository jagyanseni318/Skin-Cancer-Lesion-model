"""Sequential driver: train -> evaluate for each seed, then a longer-schedule convergence check (seed 42, 100 epochs).
Skips any step whose output already exists, so it can be restarted."""
import subprocess, sys, os
from pathlib import Path
here = Path(__file__).resolve().parent
OUTM = here / 'out' / 'models'


def run(args, log):
    print('RUN', ' '.join(args), flush=True)
    with open(here / log, 'a') as f:
        r = subprocess.run([sys.executable] + args, cwd=here, stdout=f, stderr=subprocess.STDOUT)
    print('exit', r.returncode, flush=True)
    return r.returncode


for seed in [42, 43, 44, 45, 46]:
    if not (OUTM / f'seed{seed}_summary.json').exists():
        run(['train.py', '--seed', str(seed), '--epochs', '60', '--workers', '4'], f'train{seed}.log')
    if not (here / 'out' / f'seed{seed}' / 'eval_summary.json').exists():
        run(['eval_seed.py', '--seed', str(seed)] + ([] if seed == 42 else ['--baselines', 'mean']), f'eval{seed}.log')
# convergence sensitivity: longer (100-epoch) fully-annealed schedule, seed 42
if not (OUTM / 'seed42_ep100_summary.json').exists():
    run(['train.py', '--seed', '42', '--epochs', '100', '--workers', '4', '--tag', '_ep100'], 'train42_ep100.log')
if not (here / 'out' / 'seed42_ep100' / 'eval_summary.json').exists():
    run(['eval_seed.py', '--seed', '42', '--tag', '_ep100', '--skip_faith'], 'eval42_ep100.log')
print('ALL DONE', flush=True)

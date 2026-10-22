#!/usr/bin/env python3

import os as _os, sys as _sys
_f = _os.path.abspath(__file__)
_here = _os.path.dirname(_f)
_pkg = _os.path.dirname(_here) if _os.path.basename(_here) in ('preprocessing','scoring','figures') else _here
_os.environ.setdefault('ASSIMCAST_ROOT', _pkg)
for _p in (_pkg, _os.path.join(_pkg,'model'), _os.path.join(_pkg,'preprocessing'),
           _os.path.join(_pkg,'scoring'), _os.path.join(_pkg,'figures')):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

"""
bootstrap_ci.py — confidence intervals and paired tests for the 2024 verification table.

The table reports point estimates. A Nature-family reviewer will ask how much of each
difference is sampling noise, and with only 40 events that question is real. This resamples
the events (the unit of independence is the storm, not the grid point) and reports a
percentile interval, plus a paired test between the model and its benchmarks because the
same events are scored by every method.

Reads results/final_eval.json and results/ensemble_eval_k8.json.
Writes results/bootstrap_ci.json and prints a table.
"""
import json
import os
import sys

import numpy as np

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
B = 10000
RNG = np.random.default_rng(20240924)


def ci(v, b=B):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size < 3:
        return (float('nan'), float('nan'), float('nan'), int(v.size))
    idx = RNG.integers(0, v.size, (b, v.size))
    m = v[idx].mean(axis=1)
    return (float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(v.size))


def paired(a, b):
    """Paired comparison on the events both methods scored."""
    a = np.asarray([np.nan if x is None else x for x in a], float)
    b = np.asarray([np.nan if x is None else x for x in b], float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 5:
        return dict(n=int(a.size))
    d = a - b
    idx = RNG.integers(0, d.size, (B, d.size))
    dm = d[idx].mean(axis=1)
    # two-sided bootstrap p: how often does the resampled mean change sign
    p = 2 * min((dm <= 0).mean(), (dm >= 0).mean())
    return dict(n=int(d.size), mean_diff=float(d.mean()),
                lo=float(np.percentile(dm, 2.5)), hi=float(np.percentile(dm, 97.5)),
                p=float(min(1.0, p)))


def main():
    d = json.load(open(f'{HERE}/results/final_eval.json'))
    pe = d['per_event']
    # the event is the independent unit; weight each storm equally rather than pooling points
    cols = ['base_rmse', 'gnssr_rmse', 'all_rmse', 'base_b15', 'all_b15', 'base_b20', 'all_b20',
            'base_b25', 'all_b25', 'base_ho', 'all_ho', 'buoy_base', 'buoy_all', 'buoy_pers',
            'buoy_ecmwf']
    out = {'n_events': len(pe), 'resamples': B, 'ci': {}, 'paired': {}}
    print(f'n = {len(pe)} events, {B} bootstrap resamples\n')
    print(f"{'metric':<26}{'mean':>9}{'95% CI':>20}{'n':>5}")
    for c in cols:
        v = [r.get(c) for r in pe]
        m, lo, hi, n = ci(v)
        out['ci'][c] = dict(mean=m, lo=lo, hi=hi, n=n)
        print(f'{c:<26}{m:9.3f}   [{lo:7.3f}, {hi:7.3f}]{n:5d}')

    print(f"\n{'paired comparison':<34}{'mean diff':>10}{'95% CI':>20}{'p':>9}")
    pairs = [('all_rmse', 'base_rmse', 'assimilation vs none (ERA5)'),
             ('all_b20', 'base_b20', 'assimilation vs none (>20)'),
             ('all_b25', 'base_b25', 'assimilation vs none (>25)'),
             ('all_ho', 'base_ho', 'assimilation vs none (held-out)'),
             ('buoy_all', 'buoy_pers', 'model vs persistence (buoys)'),
             ('buoy_ecmwf', 'buoy_pers', 'IFS-HRES vs persistence (buoys)')]
    for a, b, lab in pairs:
        r = paired([x.get(a) for x in pe], [x.get(b) for x in pe])
        out['paired'][lab] = r
        if r.get('p') is None:
            print(f'{lab:<34}{"(too few events)":>26}'); continue
        star = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'n.s.'
        print(f"{lab:<34}{r['mean_diff']:10.3f}   [{r['lo']:7.3f}, {r['hi']:7.3f}]{r['p']:9.4f}  {star}")

    json.dump(out, open(f'{HERE}/results/bootstrap_ci.json', 'w'), indent=1)
    print('\nsaved results/bootstrap_ci.json')


if __name__ == '__main__':
    main()

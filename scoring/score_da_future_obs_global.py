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

# score_da_future_obs_global.py
# Global (all-basin) closed-loop, leakage-free test of training-free DPS assimilation.
#
# Same protocol as score_da_future_obs.py, but loops over every basin produced by
# predict_v6_da.py (data/hafs_pred_v7_da_global/pred_{BASIN}_{da,noda}.npz +
# meta_{BASIN}_{da,noda}.csv).  At t1 = tau+6h we score against independent future
# overpasses in [t1-H, t1+H] (default H=3h); the assimilation window ends at tau, so
# the two windows are separated by >=3h and a polar-orbiting pass cannot be in both.
# DA and no-DA share identical initial fields and seeds.
#
# Primary significance test is at the EVENT (storm, 6h) level because pixels within a
# storm are not independent: equal-weight event bootstrap, point-pooled event
# bootstrap, and a two-sided sign test, mirroring the west-Pacific pipeline.
#
# Outputs: results/da_future_obs_global.{json,csv}
import os, json, glob, argparse, pickle
import numpy as np
import pandas as pd

HERE = os.environ.get("ASSIMCAST_ROOT", os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--da_dir", default=os.path.join(HERE, "data/hafs_pred_v7_da_global"))
ap.add_argument("--manifest", default=os.path.join(HERE, "data/da_v6/causal_manifest_global.pkl"))
ap.add_argument("--res", type=float, default=0.06)
ap.add_argument("--size", type=int, default=336)
ap.add_argument("--future_hr", type=float, default=3.0)
ap.add_argument("--strict_hr", type=float, default=0.5)
ap.add_argument("--spatial_deg", type=float, default=10.0)
ap.add_argument("--tag", default="da_future_obs_global")
args = ap.parse_args()
C = (args.size - 1) / 2.0
BASINS = ["WP", "NA", "SI", "SP", "CP", "EP", "NI"]
WBINS = [3, 17.5, 25.7, 33, 50, 100]
WLBL = ["3-17.5", "17.5-25.7", "25.7-33", "33-50", ">=50"]


def parse(h):
    return pd.to_datetime(str(int(h)), format="%Y%m%d%H")


def load_obs():
    o = {}
    for s in ["sar", "smap", "amsr2"]:
        z = np.load(os.path.join(HERE, f"data/observations/compat/{s}_global_30min.npz"),
                    allow_pickle=True)
        o[s] = dict(t=pd.to_datetime(z["time_ns"]),
                    lat=z["lat"].astype(float), lon=z["lon"].astype(float),
                    ws=z["ws"].astype(float))
    return o


def interp_ws(U, V, lat0, lon0, qlat, qlon):
    fi = (qlat - lat0) / args.res
    fj = (qlon - lon0) / args.res
    i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int)
    ok = (i0 >= 0) & (i0 < args.size - 1) & (j0 >= 0) & (j0 < args.size - 1)
    out = np.full(qlat.shape, np.nan)
    if not ok.any():
        return out
    ai = (fi - i0)[ok]; aj = (fj - j0)[ok]; i0 = i0[ok]; j0 = j0[ok]

    def bl(F):
        return ((1 - ai) * (1 - aj) * F[i0, j0] + ai * (1 - aj) * F[i0 + 1, j0]
                + (1 - ai) * aj * F[i0, j0 + 1] + ai * aj * F[i0 + 1, j0 + 1])

    out[ok] = np.hypot(bl(U), bl(V))
    return out


def rms_bias_corr(y, p):
    k = np.isfinite(y) & np.isfinite(p)
    y, p = y[k], p[k]
    if len(y) < 5:
        return None
    err = p - y
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean(err ** 2))),
                bias=float(np.mean(err)),
                corr=float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan)


def boot_rmse(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        out.append(np.sqrt(np.mean((p[idx] - y[idx]) ** 2)))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_boot_improvement(y, p_da, p_noda, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(y); diff = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        rn = np.sqrt(np.mean((p_noda[idx] - y[idx]) ** 2))
        rd = np.sqrt(np.mean((p_da[idx] - y[idx]) ** 2))
        diff.append(rn - rd)
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def sign_test_p(delta):
    """Two-sided exact sign test: positive vs negative event deltas (zeros dropped)."""
    d = np.asarray(delta, float)
    d = d[np.abs(d) > 1e-9]
    n = len(d)
    if n == 0:
        return np.nan
    k = int((d > 0).sum())
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        from math import comb
        ks = min(k, n - k)
        p_one = sum(comb(n, i) for i in range(ks + 1)) / 2 ** n
        return float(min(1.0, 2 * p_one))


def event_statistics(df, B=2000, seed=7):
    """Event-level statistics (primary).  Event = (sid, t1).  Vectorised bootstrap."""
    if len(df) == 0:
        return None
    t = df.copy()
    t["se_da"] = (t["da"].values - t["y"].values) ** 2
    t["se_no"] = (t["noda"].values - t["y"].values) ** 2
    ev = t.groupby(["sid", "t1"]).agg(
        basin=("basin", "first"), n=("y", "size"),
        sse_da=("se_da", "sum"), sse_no=("se_no", "sum")).reset_index()
    ev["da_rmse"] = np.sqrt(ev["sse_da"] / ev["n"])
    ev["noda_rmse"] = np.sqrt(ev["sse_no"] / ev["n"])
    ev["delta"] = ev["noda_rmse"] - ev["da_rmse"]
    delta = ev["delta"].values
    n_ev = len(ev)
    sse_da = ev["sse_da"].values; sse_no = ev["sse_no"].values; nj = ev["n"].values
    rng = np.random.default_rng(seed)
    # equal-weight event bootstrap: resample events, average per-event RMSE delta
    idx = rng.integers(0, n_ev, size=(B, n_ev))
    eq = delta[idx].mean(axis=1)
    # point-pooled event bootstrap: resample events (multinomial counts), pool their
    # squared errors, recompute pooled RMSE delta — fully vectorised, no concat.
    counts = rng.multinomial(n_ev, np.ones(n_ev) / n_ev, size=B)
    n_b = counts @ nj
    pp = np.sqrt((counts @ sse_no) / n_b) - np.sqrt((counts @ sse_da) / n_b)
    pooled_da = float(np.sqrt(sse_da.sum() / nj.sum()))
    pooled_no = float(np.sqrt(sse_no.sum() / nj.sum()))
    npos = int((delta > 0).sum())
    ev_names = [f"{s}_{t1}" for s, t1 in zip(ev["sid"], ev["t1"])]
    return dict(n_events=int(n_ev),
                events=ev_names,
                event_n=nj.astype(int).tolist(),
                event_da_rmse=[float(x) for x in ev["da_rmse"]],
                event_noda_rmse=[float(x) for x in ev["noda_rmse"]],
                event_delta=[float(x) for x in delta],
                events_DA_helps=npos, events_DA_helps_frac=float(npos / n_ev),
                pooled_da_rmse=pooled_da, pooled_noda_rmse=pooled_no,
                pooled_delta=pooled_no - pooled_da,
                equal_weight_mean_delta=float(delta.mean()),
                equal_weight_median_delta=float(np.median(delta)),
                equal_weight_event_CI=[float(np.percentile(eq, 2.5)), float(np.percentile(eq, 97.5))],
                pointpooled_event_CI=[float(np.percentile(pp, 2.5)), float(np.percentile(pp, 97.5))],
                sign_test_p=sign_test_p(delta))


def point_block(dd):
    d = {}
    for label, col in [("DA", "da"), ("no-DA", "noda")]:
        st = rms_bias_corr(dd["y"], dd[col])
        if st:
            st["rmse_ci"] = boot_rmse(dd["y"].values, dd[col].values)
            d[label] = st
    lo, hi = paired_boot_improvement(dd["y"].values, dd["da"].values, dd["noda"].values)
    d["RMSE_noda_minus_DA_CI"] = [lo, hi]
    d["DA_helps"] = bool(lo > 0)
    return d


def main():
    obs = load_obs()
    # assimilated sources per sample (for cross-source subset definition)
    entries = pickle.load(open(args.manifest, "rb"))
    assim_src = {}
    for e in entries:
        assim_src.setdefault((e["basin"], e["sid"], str(int(e["tau"]))), set()).add(e["src"])

    rec = []
    nbasin_loaded = 0
    for B in BASINS:
        fp = os.path.join(args.da_dir, f"pred_{B}_da.npz")
        mp = os.path.join(args.da_dir, f"meta_{B}_da.csv")
        if not os.path.exists(fp):
            continue
        meta = pd.read_csv(mp, keep_default_na=False)
        zda = np.load(fp)["f"]; zno = np.load(os.path.join(args.da_dir, f"pred_{B}_noda.npz"))["f"]
        nbasin_loaded += 1
        print(f"[{B}] {len(meta)} cat3+ samples", flush=True)
        for k, mt in meta.iterrows():
            clat = float(mt["clat"]); clon = float(mt["clon"])
            t1 = parse(mt["t1"])
            key = (B, mt["sid"], str(int(mt["tau"])))
            asrc = assim_src.get(key, set())
            lat1d = clat + (np.arange(args.size) - C) * args.res
            lon1d = clon + (np.arange(args.size) - C) * args.res
            Uda, Vda = zda[k, 0], zda[k, 1]
            Uno, Vno = zno[k, 0], zno[k, 1]
            for s, o in obs.items():
                dt = (o["t"] - t1).total_seconds() / 3600.0
                dlat = np.abs(o["lat"] - clat)
                dlon = np.abs((o["lon"] - clon + 180) % 360 - 180)
                sel = (np.abs(dt) <= args.future_hr) & (dlat < args.spatial_deg) & (dlon < args.spatial_deg)
                if not sel.any():
                    continue
                qlat = o["lat"][sel]; qlon = o["lon"][sel]; y = o["ws"][sel]; dtt = dt[sel]
                # align obs longitude to the (clon-centred) grid frame across the
                # antimeridian / mixed 0..360 vs -180..180 conventions
                qlon = clon + ((qlon - clon + 180.0) % 360.0 - 180.0)
                wda = interp_ws(Uda, Vda, lat1d[0], lon1d[0], qlat, qlon)
                wno = interp_ws(Uno, Vno, lat1d[0], lon1d[0], qlat, qlon)
                rad = np.hypot(qlat - clat, (qlon - clon) * np.cos(np.radians(clat)))
                ok = np.isfinite(wda) & np.isfinite(wno)
                for i in np.where(ok)[0]:
                    rec.append(dict(basin=B, sid=mt["sid"], year=str(mt["t1"])[:4],
                                    t1=str(mt["t1"]), src=s,
                                    cross_source=int(s not in asrc), dt_hr=float(dtt[i]),
                                    rad=float(rad[i]),
                                    lat=float(qlat[i]), lon=float(qlon[i]),
                                    clat=clat, clon=clon,
                                    y=float(y[i]), da=float(wda[i]), noda=float(wno[i])))
    df = pd.DataFrame(rec)
    if len(df) == 0:
        print("no independent future observations matched across basins")
        return
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", f"{args.tag}.csv"), index=False)

    def summarize(dd):
        d = dict(n_points=int(len(dd)), n_events=int(dd.groupby(["sid", "t1"]).ngroups))
        d.update(point_block(dd))
        d["event_level"] = event_statistics(dd)
        return d

    out = dict(future_hr=args.future_hr, strict_hr=args.strict_hr,
               n_basins_loaded=nbasin_loaded,
               n_points=int(len(df)), n_events=int(df.groupby(["sid", "t1"]).ngroups),
               basin_counts={b: int((df["basin"] == b).sum()) for b in df["basin"].unique()})

    # primary: causal 3h all sources
    out["causal_3h"] = summarize(df)
    # point-level wind bins
    out["wind_bins"] = {}
    for i, lb in enumerate(WLBL):
        dd = df[(df["y"] >= WBINS[i]) & (df["y"] < WBINS[i + 1])]
        if len(dd) >= 10:
            out["wind_bins"][lb] = point_block(dd)
    # point-level by source
    out["by_source"] = {s: point_block(dd) for s, dd in df.groupby("src") if len(dd) >= 10}
    # by basin (point + event)
    out["by_basin"] = {}
    for b in BASINS:
        dd = df[df["basin"] == b]
        if len(dd) >= 10:
            out["by_basin"][b] = summarize(dd)
    # cross-source subset (verification source differs from every assimilated source)
    dc = df[df["cross_source"] == 1]
    if len(dc) >= 10:
        out["cross_source"] = summarize(dc)
    # strict half-hour window (diagnostic)
    ds = df[df["dt_hr"].abs() <= args.strict_hr]
    if len(ds) >= 10:
        out["strict_30min"] = summarize(ds)
    # SAR high-wind emphasis (>=33 m/s)
    dh = df[(df["src"] == "sar") & (df["y"] >= 33)]
    if len(dh) >= 5:
        out["sar_highwind"] = summarize(dh)

    with open(os.path.join(HERE, "results", f"{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    # console
    c = out["causal_3h"]; ev = c["event_level"]
    print(f"\n=== GLOBAL independent future overpass verification (t1 +/-{args.future_hr}h) ===")
    print(f"basins loaded={nbasin_loaded}  points={out['n_points']}  events={out['n_events']}")
    print(f"  basin point counts: {out['basin_counts']}")
    for lab in ("DA", "no-DA"):
        st = c[lab]
        print(f"  {lab:5s} pooled RMSE={st['rmse']:5.2f} CI={np.round(st['rmse_ci'],2)} "
              f"bias={st['bias']:+5.2f} r={st['corr']:.3f}")
    print(f"  point-paired improvement CI={np.round(c['RMSE_noda_minus_DA_CI'],3)}")
    print(f"  EVENT: {ev['events_DA_helps']}/{ev['n_events']} positive; "
          f"equal-weight mean delta={ev['equal_weight_mean_delta']:.3f} "
          f"CI={np.round(ev['equal_weight_event_CI'],3)}; sign-test p={ev['sign_test_p']:.4f}")
    for name in ("cross_source", "strict_30min"):
        if name in out:
            e = out[name]["event_level"]
            print(f"  [{name}] pts={out[name]['n_points']} events={e['n_events']} "
                  f"DA={e['pooled_da_rmse']:.2f} noDA={e['pooled_noda_rmse']:.2f} "
                  f"{e['events_DA_helps']}/{e['n_events']} pos p={e['sign_test_p']:.4f}")
    print("  by basin (pooled RMSE DA/noDA, events):")
    for b, d in out["by_basin"].items():
        e = d["event_level"]
        print(f"    {b:3s} n={d['n_points']:5d} ev={e['n_events']:3d} "
              f"DA={d['DA']['rmse']:5.2f} noDA={d['no-DA']['rmse']:5.2f} "
              f"{e['events_DA_helps']}/{e['n_events']}")
    print("wrote results/{}.json/.csv".format(args.tag))


if __name__ == "__main__":
    main()

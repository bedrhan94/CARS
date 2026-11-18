"""Phase 10B analysis: the frozen primary statistic and decision rule (PHASE10B_PREREGISTRATION_FINAL.md sections 5-7),
followed by secondary, non-decisive descriptions. Written and frozen BEFORE any held-out input was generated.

Primary: mode ST, size S3, arms PRIMARY[f] = (per-system P, reuse Q):
  per (family, seed, repeat): R = measured certified cost of P / that of Q (same process);
  per unit (family, seed): median of R over the 3 repeats;   per family: GM over the 5 units,
  two-sided 95 % t-interval on log R (df = 4); 90 % t-interval for the equivalence description.
Parity: a run (all 21 records of one arm in one cell) is parity-valid iff all records are CERTIFIED, target-met, and
VERIFIED on replay. Speed is decisive for a family only if both primary arms are parity-valid in every run.
Per-family category: INVALID (hard stop, or per-system arm parity failure); REUSE ACCURACY-INFERIOR (reuse arm parity
failure); REUSE ADVANTAGE (GM >= 1.2 and lower 95 % > 1); PER-SYSTEM ADVANTAGE (GM <= 1/1.2 and upper 95 % < 1);
otherwise NO ESTABLISHED PRACTICAL ADVANTAGE. Joint outcome: equal categories -> that category; any INVALID -> INVALID;
otherwise MIXED. Descriptors (not decisive): 'directional' if the 95 % interval excludes 1; 'practically equivalent'
if the 90 % interval lies inside (1/1.2, 1.2) (TOST, alpha 0.05 per side).
Usage: python analyze10b.py [runs_dir] [out_json]"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
T950 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132}
MARGIN = 1.2


def gm_ci(vals):
    l = np.log(np.asarray(vals, float)); m = float(l.mean()); n = len(l)
    if n < 2:
        return dict(gm=math.exp(m), n=n, ci95=None, ci90=None)
    se = float(l.std(ddof=1)) / math.sqrt(n)
    return dict(gm=math.exp(m), n=n, ci95=[math.exp(m - T975[n - 1] * se), math.exp(m + T975[n - 1] * se)],
                ci90=[math.exp(m - T950[n - 1] * se), math.exp(m + T950[n - 1] * se)], sd_log=float(l.std(ddof=1)))


def load(runs):
    """cells[(mode, size, family, seed, repeat)] = {arm: summary + parity}; first attempt unless a retry exists."""
    cells, replays, envs = {}, {}, {}
    for stage in ('ST_S3', 'ST_S2', 'MT_S3', 'MT_S2'):
        for cj in sorted((runs / stage).glob('*/cell_*.json')):
            cdir = cj.parent
            d = json.loads(cj.read_text()); env = d['env']; res = d['results']
            r0 = res[0]; key = (r0['mode'], r0['size'], r0['family'], r0['seed'], r0['repeat'])
            if key in cells and not cdir.name.endswith('_retry'):
                continue
            rp = json.loads((cdir / 'replay10b.json').read_text()) if (cdir / 'replay10b.json').exists() else None
            arms = {}
            for s in res:
                ra = (rp or {}).get('arms', {}).get(s['arm'], {})
                verified = ra.get('VERIFIED', 0)
                s = dict(s, replay_verified=verified,
                         parity=(s['records'] == 21 and s['certified_target'] == 21 and verified == 21))
                arms[s['arm']] = s
            cells[key] = arms; replays[key] = rp; envs[key] = env
    return cells, replays, envs


def primary(cells, mode='ST', size='S3'):
    out = {}
    for fam, (P, Q) in K.PRIMARY.items():
        runs_ = {k: v for k, v in cells.items() if k[0] == mode and k[1] == size and k[2] == fam}
        per_seed = defaultdict(list); parity_fail = {P: [], Q: []}; missing = []
        for seed in K.HELDOUT_SEEDS:
            for rep in (0, 1, 2):
                c = runs_.get((mode, size, fam, seed, rep))
                if c is None or P not in c or Q not in c:
                    missing.append((seed, rep)); continue
                for arm in (P, Q):
                    if not c[arm]['parity']:
                        parity_fail[arm].append((seed, rep))
                per_seed[seed].append(c[P]['measured_s'] / c[Q]['measured_s'])
        units = {s: float(np.median(v)) for s, v in per_seed.items()}
        st = gm_ci(list(units.values())) if units else None
        if missing:
            cat = 'INVALID'; why = f'missing runs {missing}'
        elif parity_fail[P]:
            cat = 'INVALID'; why = f'per-system arm {P} parity failure {parity_fail[P]}'
        elif parity_fail[Q]:
            cat = 'REUSE ACCURACY-INFERIOR'; why = f'reuse arm {Q} parity failure {parity_fail[Q]}'
        elif st['gm'] >= MARGIN and st['ci95'][0] > 1.0:
            cat = 'REUSE ADVANTAGE'; why = 'GM >= 1.2 and lower 95 % bound > 1'
        elif st['gm'] <= 1 / MARGIN and st['ci95'][1] < 1.0:
            cat = 'PER-SYSTEM ADVANTAGE'; why = 'GM <= 1/1.2 and upper 95 % bound < 1'
        else:
            cat = 'NO ESTABLISHED PRACTICAL ADVANTAGE'; why = 'neither directional threshold met'
        desc = {}
        if st and st['ci95']:
            desc['directional_95'] = ('per-system faster' if st['ci95'][1] < 1 else 'reuse faster' if st['ci95'][0] > 1
                                      else 'interval includes 1')
            desc['practically_equivalent_tost90'] = bool(st['ci90'][0] > 1 / MARGIN and st['ci90'][1] < MARGIN)
        out[fam] = dict(per_system=P, reuse=Q, per_seed_repeat={str(s): v for s, v in per_seed.items()},
                        per_seed_median={str(s): v for s, v in units.items()}, stats=st, category=cat, reason=why,
                        parity_failures={k: v for k, v in parity_fail.items()}, missing=missing, descriptors=desc)
    cats = [v['category'] for v in out.values()]
    if 'INVALID' in cats:
        joint = 'INVALID'
    elif len(set(cats)) == 1 and cats[0] in ('REUSE ADVANTAGE', 'PER-SYSTEM ADVANTAGE', 'NO ESTABLISHED PRACTICAL ADVANTAGE'):
        joint = cats[0]
    else:
        joint = 'MIXED'
    return out, joint


def secondary(cells, envs):
    """Per (mode, size, family): every arm's cost (per-seed median over repeats, mean over seeds), parity counts, the
    primary-pair ratio statistics, counterfactual-B ratio, end-to-end (incl. I/O) ratio, all pairwise ratios against
    each reuse arm, repeat CV, components, PCG iterations, EFT calls, factorizations, peak memory."""
    out = {}
    groups = defaultdict(dict)
    for k, c in cells.items():
        groups[k[:3]][(k[3], k[4])] = c
    for (mode, size, fam), runs_ in sorted(groups.items()):
        arms = sorted({a for c in runs_.values() for a in c})
        g = dict(runs=len(runs_), arms={})
        for a in arms:
            by_seed = defaultdict(list); cf = defaultdict(list); e2e = defaultdict(list)
            comps = defaultdict(list); it = []; eft = []; fac = []; par = 0; tot = 0; peak = []
            for (seed, rep), c in runs_.items():
                s = c.get(a)
                if not s:
                    continue
                tot += 1; par += int(s['parity'])
                by_seed[seed].append(s['measured_s']); cf[seed].append(s['counterfactual_s'])
                e2e[seed].append(s['measured_s'] + s['io_s'])
                for ck, cv in s['components'].items():
                    comps[ck].append(cv)
                it.append(s['pcg_iters']); eft.append(s.get('eft_calls', 0)); fac.append(s['factorizations'])
                peak.append(s['peak_wset_GiB'])
            med = {sd: float(np.median(v)) for sd, v in by_seed.items()}
            cvs = [float(np.std(v, ddof=1) / np.mean(v)) for v in by_seed.values() if len(v) > 1]
            g['arms'][a] = dict(cost_mean_of_seed_medians=float(np.mean(list(med.values()))), per_seed_median=med,
                                parity_runs=f'{par}/{tot}', certified_target=sum(c[a]['certified_target'] for c in runs_.values() if a in c),
                                records=sum(c[a]['records'] for c in runs_.values() if a in c),
                                counterfactual_mean=float(np.mean([np.median(v) for v in cf.values()])),
                                end_to_end_mean=float(np.mean([np.median(v) for v in e2e.values()])),
                                repeat_cv_median=(float(np.median(cvs)) if cvs else None), repeat_cv_max=(max(cvs) if cvs else None),
                                components_mean_per_run={ck: float(np.mean(cv)) for ck, cv in comps.items()},
                                pcg_iters_mean=float(np.mean(it)), eft_calls_mean=float(np.mean(eft)),
                                factorizations_mean=float(np.mean(fac)), peak_wset_GiB_max=max(peak))
        pairs = {}
        for P in ('A', 'B1', 'B', 'B9'):
            for Q in ('C', 'D', 'D2'):
                for kind, field in (('measured', 'measured_s'), ('counterfactual', 'counterfactual_s'), ('end_to_end', None)):
                    per = defaultdict(list)
                    for (seed, rep), c in runs_.items():
                        if P in c and Q in c:
                            if field:
                                per[seed].append(c[P][field] / c[Q][field])
                            else:
                                per[seed].append((c[P]['measured_s'] + c[P]['io_s']) / (c[Q]['measured_s'] + c[Q]['io_s']))
                    if per:
                        units = {sd: float(np.median(v)) for sd, v in per.items()}
                        pairs[f'{P}/{Q}|{kind}'] = dict(per_seed=units, **gm_ci(list(units.values())))
        g['pairs'] = pairs
        out[f'{mode}|{size}|{fam}'] = g
    return out


def validity(cells, replays):
    tot = defaultdict(int); errors = 0; mism = 0; reproofs = defaultdict(int); views = defaultdict(lambda: defaultdict(set))
    for k, rp in replays.items():
        if rp is None:
            tot['NO_REPLAY'] += 1; continue
        for s, n in rp['totals'].items():
            tot[s] += n
        errors += len(rp['errors']); mism += len(rp['manifest_mismatches'])
        for s, n in rp['rump_reproofs'].items():
            reproofs[s] += n
        for vk, reps in rp.get('certificate_views', {}).items():
            for rep, h in reps.items():
                views[(k[0],) + tuple(vk.split('|'))][rep].add(h)
    det = {}
    for vk, reps in views.items():
        hs = set().union(*reps.values())
        det['|'.join(map(str, vk))] = dict(repeats=len(reps), identical=(len(hs) == 1))
    return dict(replay_totals=dict(tot), replay_errors=errors, manifest_mismatches=mism, rump_reproofs=dict(reproofs),
                determinism_identical=sum(d['identical'] for d in det.values()), determinism_groups=len(det),
                determinism_nonidentical=[k for k, d in det.items() if not d['identical']])


def session2(runs, cells):
    """Amendment A10: same-machine second session (stage R2_ST_S3, repeat index 3). Per family: the primary ratio per
    seed in session 2, its GM and 95 % interval, and the per-seed log difference from the session-1 medians."""
    out = {}
    s2 = {}
    for cj in sorted((runs / 'R2_ST_S3').glob('*/cell_*.json')):
        d = json.loads(cj.read_text()); rp = cj.parent / 'replay10b.json'
        rpd = json.loads(rp.read_text()) if rp.exists() else None
        for s in d['results']:
            ra = (rpd or {}).get('arms', {}).get(s['arm'], {})
            s2.setdefault((s['family'], s['seed']), {})[s['arm']] = dict(s, parity=(s['records'] == 21 and s['certified_target'] == 21
                                                                                    and ra.get('VERIFIED', 0) == 21))
    if not s2:
        return None
    s1, _ = primary(cells)
    for fam, (P, Q) in K.PRIMARY.items():
        per = {sd: c[P]['measured_s'] / c[Q]['measured_s'] for (f, sd), c in s2.items() if f == fam and P in c and Q in c}
        par = all(c[a]['parity'] for (f, sd), c in s2.items() if f == fam for a in (P, Q) if a in c)
        diff = {str(sd): math.log(r / s1[fam]['per_seed_median'][str(sd)]) for sd, r in per.items()
                if str(sd) in s1[fam]['per_seed_median']}
        out[fam] = dict(per_seed={str(k): v for k, v in per.items()}, stats=(gm_ci(list(per.values())) if per else None),
                        parity_all=par, log_diff_vs_session1=diff,
                        arm_costs={a: float(np.mean([c[a]['measured_s'] for (f, sd), c in s2.items() if f == fam and a in c]))
                                   for a in [x[0] for x in K.ARMS]})
    return out


def main():
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'runs'
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'analysis10b.json'
    cells, replays, envs = load(runs)
    prim, joint = primary(cells)
    res = dict(primary=prim, joint_outcome=joint, secondary_mt_s3=primary(cells, 'MT', 'S3'),
               secondary_st_s2=primary(cells, 'ST', 'S2'), secondary_mt_s2=primary(cells, 'MT', 'S2'),
               descriptive=secondary(cells, envs), validity=validity(cells, replays), session2=session2(runs, cells),
               cells=len(cells), env_summary={'|'.join(map(str, k)): dict(mode=e['mode'], mkl_max_threads=e['mkl_max_threads'],
                                                                          affinity=e['affinity'], fp_before=e['fingerprint_before'],
                                                                          fp_after=e.get('fingerprint_after'),
                                                                          peak_wset_GiB=e.get('peak_wset_GiB'),
                                                                          input_check=e['input_check'])
                                              for k, e in envs.items()})
    outp.write_text(json.dumps(res, indent=1, default=str))
    print('JOINT OUTCOME (ST S3, primary):', joint)
    for f, v in prim.items():
        st = v['stats'] or {}
        print(f, v['per_system'], '/', v['reuse'], v['category'], 'GM', st.get('gm'), 'CI95', st.get('ci95'), v['descriptors'])
    print('validity', json.dumps(res['validity'])[:600])


if __name__ == '__main__':
    main()

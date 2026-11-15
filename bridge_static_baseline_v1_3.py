"""Comparator run for the cross-hospital study: the paper's moving patient reference against
(a) a fixed per-subject median, (b) a location-scale version, and (c) other baseline lengths.

Detectors (18 channels; everything else - features, network, training, thresholds from the
training hospital, SzCORE scoring - unchanged):
    BRIDGE-R      features minus the running median of the previous 10 min (the paper's reference)
    BRIDGE-S      features minus ONE fixed median per subject (a fixed offset, not causal)
    BRIDGE-RS     (features minus running median) / running IQR of the same 10 min (location-scale, causal)
    BRIDGE-R2.5 / BRIDGE-R5 / BRIDGE-R20   running median over 2.5, 5 or 20 min instead of 10
    GBDT-R, GBDT-S, GBDT-RS                gradient-boosted trees with the same three references

Stages
    static      seed 0, all four settings: BRIDGE-R, BRIDGE-S, GBDT-R, GBDT-S
    scale       seed 0, all four settings: BRIDGE-RS, GBDT-RS
    wsweep      seed 0, both transfer directions: BRIDGE-R2.5, BRIDGE-R5, BRIDGE-R20
    cmp_seeds   seeds 1-4, both transfer directions: BRIDGE-R/S/RS and GBDT-R/S/RS
                (with seed 0 that gives five seeds per transfer comparison)

How to run (Google Colab, A100 or L4 GPU runtime; T4 works but is slower)
    %cd /content
    !test -d BRIDGE-cross-hospital || git clone https://github.com/vanithamagesh/BRIDGE-cross-hospital.git
    %cd /content/BRIDGE-cross-hospital
    !pip install -q timescoring
    # upload this file into the folder, then:
    !python bridge_static_baseline_v1_3.py --smoke     # quick pipeline check
    !python bridge_static_baseline_v1_3.py             # full run

    An existing feature cache in --data is reused; without one the EEG is downloaded again
    (about 68 GB, deleted after use). If /content/xhosp_static_out still holds results.pkl from
    the earlier comparator run, its finished configurations are kept and only the new ones train.
    Outputs in /content/xhosp_static_out: results_runs.csv, results_subjects.csv, sweeps.csv
    and comparison_summary.csv (mean and s.d. over seeds).

Notes
    The running IQR divisor is floored at 0.1 to avoid division by a near-zero spread.
    2.5, 5, 10 and 20 min correspond to 38, 75, 150 and 300 windows of 4 s.
"""
import os, sys

if __name__ != '__main__': raise SystemExit

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'bridge_cross_hospital.py')
if not os.path.exists(MAIN):
    sys.exit('Put this file in the same folder as bridge_cross_hospital.py (the BRIDGE-cross-hospital repository).')
if '--out' not in sys.argv: sys.argv += ['--out', '/content/xhosp_static_out']
if '--stop_after' not in sys.argv: sys.argv += ['--stop_after', 'cmp_seeds']
if '--no_raw' not in sys.argv: sys.argv += ['--no_raw']
src = open(MAIN, encoding='utf-8').read()


def patch(old, new):
    global src
    assert src.count(old) == 1, 'bridge_cross_hospital.py differs from the version this script was written for:\n' + old[:90]
    src = src.replace(old, new)


# 1. the comparator detectors
patch("'GBDT-R': ('gbdt', 'rel'), 'RawCNN': ('cnn', 'raw')}",
      "'GBDT-R': ('gbdt', 'rel'), 'RawCNN': ('cnn', 'raw'), 'BRIDGE-S': ('gnn', 'sta'), 'GBDT-S': ('gbdt', 'sta'),\n"
      "          'BRIDGE-RS': ('gnn', 'rs'), 'GBDT-RS': ('gbdt', 'rs'),\n"
      "          'BRIDGE-R2.5': ('gnn', 'r38'), 'BRIDGE-R5': ('gnn', 'r75'), 'BRIDGE-R20': ('gnn', 'r300')}")

# 2. the stages that are run instead of the paper's
patch("safe = lambda s: re.sub(r'[^A-Za-z0-9_.-]+', '_', s)",
      "M_CMP = ('BRIDGE-R', 'BRIDGE-RS', 'BRIDGE-S', 'BRIDGE-R2.5', 'BRIDGE-R5', 'BRIDGE-R20', 'GBDT-R', 'GBDT-RS', 'GBDT-S')\n"
      "STAGES = [('static', [(e, m, 18, 0) for e in EXPS for m in ('BRIDGE-R', 'BRIDGE-S', 'GBDT-R', 'GBDT-S')]),\n"
      "          ('scale', [(e, m, 18, 0) for e in EXPS for m in ('BRIDGE-RS', 'GBDT-RS')]),\n"
      "          ('wsweep', [(e, m, 18, 0) for e in ('x_c2s', 'x_s2c') for m in ('BRIDGE-R2.5', 'BRIDGE-R5', 'BRIDGE-R20')]),\n"
      "          ('cmp_seeds', [(e, m, 18, s) for s in (1, 2, 3, 4) for e in ('x_c2s', 'x_s2c')\n"
      "                         for m in ('BRIDGE-R', 'BRIDGE-S', 'BRIDGE-RS', 'GBDT-R', 'GBDT-S', 'GBDT-RS')])]\n"
      "safe = lambda s: re.sub(r'[^A-Za-z0-9_.-]+', '_', s)")

# 3. the extra feature blocks: fixed per-subject median, location-scale, other baseline lengths
patch("t1 = time.time(); FR = np.concatenate([Fa, Fa[..., :N_LOG] - patient_reference()], -1); del Fa",
      """def roll_block(stat, W_):
    B = np.empty((N, N_CH, N_LOG), np.float32)
    for f in range(len(FILES)):
        ii = np.where(FID == f)[0]; na = FILE_WIN[f]
        if not len(na): continue
        df = pd.DataFrame(Fa[na, :, :N_LOG].reshape(len(na), -1)).rolling(W_, min_periods=1)
        m = (df.median() if stat == 'med' else df.quantile(0.75) - df.quantile(0.25)).to_numpy(np.float32).reshape(-1, N_CH, N_LOG)
        B[ii] = m[np.clip(np.searchsorted(T0[na], T0[ii] + 1e-6, side='right') - 1, 0, len(na) - 1)]
    return B


def static_reference():
    B = np.empty((N, N_CH, N_LOG), np.float32)
    for s in range(len(SUBJ_NAMES)):
        ii = np.where(SUBJ == s)[0]; ev = ii[~AUG[ii]]
        if len(ii): B[ii] = np.median(Fa[ev if len(ev) else ii][:, :, :N_LOG], 0)
    return B


t1 = time.time()
R150 = roll_block('med', W_BASE); S150 = np.maximum(roll_block('iqr', W_BASE), 0.1)
L_ = Fa[..., :N_LOG]
FR = np.concatenate([Fa, L_ - R150, L_ - static_reference(), (L_ - R150) / S150,
                     L_ - roll_block('med', 38), L_ - roll_block('med', 75), L_ - roll_block('med', 300)], -1)
del Fa, R150, S150, L_""")

# 4. input widths and feature slicing for the new kinds
patch("N_IN = {'ref': N_F + N_LOG, 'rel': N_F, 'abs': N_F}",
      "N_IN = {'ref': N_F + N_LOG, 'rel': N_F, 'abs': N_F, 'sta': N_F, 'rs': N_F, 'r38': N_F, 'r75': N_F, 'r300': N_F}")
patch("    if kind == 'rel': x = torch.cat([x[..., N_F:], x[..., N_LOG:N_F]], -1)\n"
      "    elif kind == 'abs': x = x[..., :N_F]\n"
      "    return x, Ct[bt][:, sub.pairs_t]",
      "    OFF = {'rel': 0, 'sta': 1, 'rs': 2, 'r38': 3, 'r75': 4, 'r300': 5}\n"
      "    if kind in OFF:\n"
      "        o_ = N_F + OFF[kind] * N_LOG; x = torch.cat([x[..., o_:o_ + N_LOG], x[..., N_LOG:N_F]], -1)\n"
      "    elif kind == 'abs': x = x[..., :N_F]\n"
      "    else: x = x[..., :N_F + N_LOG]\n"
      "    return x, Ct[bt][:, sub.pairs_t]")

# 5. the summary over seeds, written before the script stops
patch("    if args.stop_after == name: sys.exit(0)",
      """    if name == 'cmp_seeds':
        rows = []
        log('')
        log('setting                 model         seeds  F1 mean     sd   sens  FA/24h   AUROC  F1 at best test threshold')
        for e in EXPS:
            for m in M_CMP:
                r = [x for x in RES['runs'] if x['exp'] == e and x['model'] == m and x['nch'] == 18 and x['op'] == 'f1']
                if not r: continue
                o = [x for x in RES['runs'] if x['exp'] == e and x['model'] == m and x['nch'] == 18 and x['op'] == 'oracle']
                mu = lambda rr, k: float(np.mean([x[k] for x in rr])); sd = lambda rr, k: float(np.std([x[k] for x in rr]))
                rows.append(dict(setting=EXP_LABEL[e], model=m, seeds=len(r), event_f1=mu(r, 'pooled_f1'), event_f1_sd=sd(r, 'pooled_f1'),
                                 sensitivity=mu(r, 'pooled_sens'), fa_per_24h=mu(r, 'pooled_fp24'), auroc=mu(r, 'auroc'),
                                 auroc_sd=sd(r, 'auroc'), f1_best_test_threshold=mu(o, 'pooled_f1')))
                log('%-23s %-13s %3d %8.3f %6.3f %6.3f %7.1f %7.3f %10.3f' % (EXP_LABEL[e], m, len(r), mu(r, 'pooled_f1'),
                    sd(r, 'pooled_f1'), mu(r, 'pooled_sens'), mu(r, 'pooled_fp24'), mu(r, 'auroc'), mu(o, 'pooled_f1')))
        write_csv(rows, 'comparison_summary.csv')
        log('saved comparison_summary.csv in ' + args.out)
    if args.stop_after == name: sys.exit(0)""")

# write the patched program next to the original and run it as an ordinary script (worker processes need a real file)
OUT = os.path.join(HERE, 'bridge_cross_hospital_cmp.py')
open(OUT, 'w', encoding='utf-8').write(src)
import subprocess
sys.exit(subprocess.call([sys.executable, OUT] + sys.argv[1:]))

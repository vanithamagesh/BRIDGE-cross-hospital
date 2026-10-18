"""BRIDGE across hospitals: cross-hospital, few-channel and streaming seizure detection on all of CHB-MIT and the Siena
Scalp EEG Database. Code of the journal paper "Seizure Detection Across Hospitals and Wearable Montages: A
Patient-Referenced Streaming Detector Evaluated on Pediatric and Adult Scalp EEG".

Run on Google Colab (Runtime > Change runtime type > A100 GPU, High-RAM; on a T4 add --no_raw, which drops the raw-signal
CNN and its 19 GB of signals) or on any machine with Python 3.10+, PyTorch and a GPU:
    python bridge_cross_hospital.py --smoke    # about 10 min: 3 subjects per hospital, small files; checks the pipeline
    python bridge_cross_hospital.py            # full run: about 68 GB download (deleted after use), about 1 h on an A100
    python bridge_cross_hospital.py --figs <results folder>    # figures only, from a results folder (no data needed)
Options: --data <download and cache folder> (default /content/xhosp_data), --out <output folder> (default
/content/xhosp_out), --workers <processes>, --no_raw, --keep_edf, --stop_after <stage>. If the run stops, start it again
with the same command: processed files, finished folds and stages are loaded from checkpoints. Outputs: log, dataset
description, result tables (CSV), figures/ (PNG at 600 dpi and PDF), xhosp_results_v1_1.zip and xhosp_preds_v1_1.zip
(window probabilities and saved models); both are copied to MyDrive if Drive is mounted. The smoke test writes to
<out>_smoke/ and its zips end in _smoke.

Data (open access, read from the PhysioNet AWS open-data mirror with physionet.org as fallback):
  CHB-MIT Scalp EEG Database v1.0.0 (Shoeb 2009; ODC-BY 1.0): all 686 recordings of 24 cases (chb21 is a later recording
  of the subject of chb01: 23 subjects, children and adolescents, Boston); bipolar montage, 256 Hz.
  Siena Scalp EEG Database v1.0.0 (Detti et al. 2020; CC BY 4.0): 41 recordings of 14 adults, Siena; referential 10-20
  EEG at 512 Hz, converted to the same 18 bipolar channels. Recording start times are read from the EDF headers; the
  seizure lists are parsed with the corrections logged in dataset.json.

Pipeline (identical offline and in the streaming implementation): causal 4th-order Butterworth band-pass 0.5-45 Hz with
the filter state carried across chunks, decimation to 256 Hz, non-overlapping 4-s windows, per channel 7 log band powers,
log line length, log variance, kurtosis and zero-crossing rate, and the inter-channel correlations; causal patient
reference: log features minus their running median over the previous 150 windows (10 min) of the same recording.
Models: BRIDGE (both views; cited from the conference paper), BRIDGE-R (referenced view only), BRIDGE-A (absolute view
only), channel Transformer, MLP, logistic regression, gradient-boosted trees (GBDT, GBDT-R) and a raw-signal 1-D CNN.
Channel sets: 18 bipolar channels, 4 temporal channels (F7-T7, T7-P7, F8-T8, T8-P8) and 2 (T7-P7, T8-P8).
Experiments: in-hospital (Siena leave-one-subject-out; CHB-MIT 4 subject-wise folds) and cross-hospital (train on all of
one database, test on all of the other). Thresholds come from inner subject-wise folds of the training subjects only:
maximum event F1, and the lowest threshold within 1, 4 and 12 false alarms per 24 h. Post-processing: alarm when 2 of the
last 3 windows are positive. Scoring: SzCORE event scoring (30 s before onset to 60 s after offset; events closer than
90 s merged; events longer than 5 min split), false alarms per 24 h, detection latency, window AUROC/AUPRC.
Further stages: hospital identifiability of the absolute and referenced features, a streaming implementation timed stage
by stage on one CPU thread (fp32 and int8) and checked against the offline pipeline, and the figures.
"""
import argparse, csv, json, math, os, pickle, re, shutil, subprocess, sys, time, urllib.parse, urllib.request, zipfile
import concurrent.futures as cf
import multiprocessing as mp
import warnings
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true', help='3 subjects per hospital, small files, 1 epoch')
ap.add_argument('--data', default='/content/xhosp_data', help='download and feature-cache folder')
ap.add_argument('--out', default='/content/xhosp_out', help='output folder')
ap.add_argument('--drive', default='/content/drive/MyDrive', help='zips are copied here if it exists')
ap.add_argument('--workers', type=int, default=0, help='parallel download/feature processes (0: automatic)')
ap.add_argument('--no_raw', action='store_true', help='skip the raw-signal CNN (saves about 19 GB of disk)')
ap.add_argument('--keep_edf', action='store_true', help='keep the EDF files after feature extraction')
ap.add_argument('--stop_after', default='', help='stop after this stage: prep, x18, in18, ch42, seeds, shift, stream')
ap.add_argument('--figs', default='', help='only redraw the figures from this results folder')
args, _ = ap.parse_known_args()
SMOKE, V = args.smoke, '1_1'
if SMOKE and args.out == ap.get_default('out'): args.out += '_smoke'   # never mixed with the full run

FS, WIN_S = 256, 4
WIN, AUG_HOP = FS * WIN_S, FS
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
CH18 = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
N_CH, N_F, N_LOG = 18, len(BANDS) + 4, len(BANDS) + 2
SIENA_EL = sorted({e for c in CH18 for e in c.split('-')})
SIENA_REN = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
SUBSETS = {18: list(range(N_CH)), 4: [CH18.index(c) for c in ('F7-T7', 'T7-P7', 'F8-T8', 'T8-P8')],
           2: [CH18.index(c) for c in ('T7-P7', 'T8-P8')]}
W_BASE = 150                                          # patient reference: previous 150 windows (10 min)
TOL_PRE, TOL_POST, MERGE_GAP, MAX_EVENT = 30, 60, 90, 300   # SzCORE event scoring (s)
K_ALARM, N_ALARM = 2, 3                               # alarm: 2 of the last 3 windows positive
BUDGETS = (1, 4, 12)                                  # false alarms per 24 h on the inner validation subjects
MIRROR = {'chb': ['https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/', 'https://physionet.org/files/chbmit/1.0.0/'],
          'siena': ['https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0/', 'https://physionet.org/files/siena-scalp-eeg/1.0.0/']}
DS_ID, DS_NAME = {'chb': 0, 'siena': 1}, {0: 'CHB-MIT', 1: 'Siena'}
SMOKE_SIENA = ['PN00/PN00-4.edf', 'PN00/PN00-5.edf', 'PN05/PN05-4.edf', 'PN12/PN12-3.edf']
EQUIV_FILE = 'PN00/PN00-4.edf' if SMOKE else 'PN00/PN00-1.edf'   # kept on disk for the streaming check
EPOCHS, EPOCHS_RAW = (1, 1) if SMOKE else (15, 10)
RATIO, BS, LR = 4, 1024, 2e-3
THR_GRID = np.unique(np.r_[np.linspace(0.05, 0.95, 19), 1 - np.logspace(-1.3, -4.5, 33)])
MODELS = {'BRIDGE': ('gnn', 'ref'), 'BRIDGE-R': ('gnn', 'rel'), 'BRIDGE-A': ('gnn', 'abs'), 'Transformer': ('tf', 'ref'),
          'MLP': ('mlp', 'ref'), 'LogReg': ('lin', 'ref'), 'GBDT': ('gbdt', 'ref'), 'GBDT-R': ('gbdt', 'rel'), 'RawCNN': ('cnn', 'raw')}
M18 = ['BRIDGE', 'BRIDGE-R', 'BRIDGE-A', 'Transformer', 'MLP', 'LogReg', 'GBDT', 'GBDT-R'] + ([] if args.no_raw else ['RawCNN'])
MSUB = ['BRIDGE', 'BRIDGE-R', 'BRIDGE-A', 'LogReg', 'GBDT']
EXPS = ['x_c2s', 'x_s2c', 'in_siena', 'in_chb']
EXP_LABEL = {'x_c2s': 'CHB-MIT → Siena', 'x_s2c': 'Siena → CHB-MIT', 'in_siena': 'Siena (in-hospital)',
             'in_chb': 'CHB-MIT (in-hospital)'}
STAGES = [('x18', [(e, m, 18, 0) for e in ('x_c2s', 'x_s2c') for m in M18]),
          ('in18', [(e, m, 18, 0) for e in ('in_siena', 'in_chb') for m in M18]),
          ('ch42', [(e, m, n, 0) for n in (4, 2) for e in EXPS for m in MSUB]),
          ('seeds', [(e, m, 18, s) for s in (1, 2) for e in ('x_c2s', 'x_s2c') for m in ('BRIDGE', 'BRIDGE-R', 'BRIDGE-A')])]
safe = lambda s: re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
T_START = time.time()


def log(*a):
    s = f'[{(time.time() - T_START) / 60:6.1f} min] ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if LOG: LOG.write(s + '\n'); LOG.flush()


LOG = None


# ================================================================== download and annotations
def fetch(ds, rel, tries=3):
    dest = os.path.join(args.data, ds, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0: return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True); err = None
    for base in MIRROR[ds]:
        for k in range(tries):
            try:
                req = urllib.request.Request(base + urllib.parse.quote(rel), headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=300) as r, open(dest + '.part', 'wb') as f:
                    n = int(r.headers.get('Content-Length', -1)); shutil.copyfileobj(r, f, 1 << 22)
                if n >= 0 and os.path.getsize(dest + '.part') != n: raise IOError('incomplete download')
                os.replace(dest + '.part', dest); return dest
            except Exception as e:
                err = e
                if getattr(e, 'code', None) in (403, 404): break
                time.sleep(2 + 4 * k)
    raise RuntimeError(f'{ds}/{rel}: {err}')


def fetch_head(ds, rel, nbytes=65536):
    """First bytes of a file (EDF header) by an HTTP range request; falls back to the full download."""
    for base in MIRROR[ds]:
        try:
            req = urllib.request.Request(base + urllib.parse.quote(rel), headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes=0-{nbytes - 1}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read(nbytes); size = int(r.headers.get('Content-Range', '/-1').split('/')[-1])
            if len(b) >= 256: return b, size
        except Exception:
            pass
    p = fetch(ds, rel); return open(p, 'rb').read(nbytes), os.path.getsize(p)


def edf_header(raw):
    ns = int(raw[252:256].decode().strip()); h = raw[256:256 + 256 * ns]
    fld = lambda off, w, typ=str: [typ(h[off * ns + i * w: off * ns + (i + 1) * w].decode('latin-1').strip()) for i in range(ns)]
    return dict(ns=ns, labels=fld(0, 16), pmin=fld(104, 8, float), pmax=fld(112, 8, float), dmin=fld(120, 8, float),
                dmax=fld(128, 8, float), nsamp=fld(216, 8, int), rec_dur=float(raw[244:252].decode().strip()),
                start=raw[176:184].decode('latin-1').strip(), hdr_bytes=256 + 256 * ns)


def parse_summary(txt):
    files, cur = {}, None
    for line in txt.splitlines():
        line = line.strip()
        m = re.match(r'File Name:\s*(\S+\.edf)', line)
        if m: cur = m.group(1); files[cur] = []; continue
        m = re.match(r'Seizure\s*\d*\s*Start Time:\s*(\d+)', line)
        if m and cur: files[cur].append([int(m.group(1)), None]); continue
        m = re.match(r'Seizure\s*\d*\s*End Time:\s*(\d+)', line)
        if m and cur and files[cur] and files[cur][-1][1] is None: files[cur][-1][1] = int(m.group(1))
    return {f: [(float(a), float(b)) for a, b in s if b is not None and b > a] for f, s in files.items()}


TIME = r'(\d{1,2})\s*[.:]\s*(\d{2})\s*[.:]\s*(\d{2})'


def hms(s):
    s = s.replace('1 6.', '16.')
    m = re.search(TIME + r'\s*\(ELECTRIC', s, re.I) or re.search(TIME, s)   # electrographic onset when both are given
    return None if m is None else int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def parse_siena(txt, pn, edfs, notes):
    """edfs: {file: (start s of day from the EDF header, duration s)} -> [(file, onset s, offset s)]."""
    out, cur, st = [], None, None

    def resolve(name):
        name = name.strip().replace('PNO', 'PN0')
        if name in edfs: return name
        cand = [f for f in edfs if f.startswith(name.replace('.edf', '').rstrip('-'))]
        if len(cand) == 1: notes.append(f'{pn}: file name {name} read as {cand[0]}'); return cand[0]
        if len(edfs) == 1: notes.append(f'{pn}: file name {name} read as {list(edfs)[0]}'); return list(edfs)[0]
        raise ValueError(f'{pn}: cannot resolve the file name {name}')
    for line in txt.replace('\r', '').splitlines():
        l = line.strip(); low = l.lower()
        if low.startswith('file name'): cur = resolve(l.split(':', 1)[1]); continue
        if re.match(r'(seizure\s+)?start\s+time', low): st = hms(l.split(':', 1)[1]); continue
        if re.match(r'(seizure\s+)?end\s+time', low) and st is not None and cur is not None:
            en = hms(l.split(':', 1)[1]); t0, dur = edfs[cur]; a, b = (st - t0) % 86400, (en - t0) % 86400
            if not (0 < b - a <= 600) and 0 < b - 3600 - a <= 600:
                notes.append(f'{pn} {cur}: seizure end {b:.0f} s reduced by 1 h (the seizure would last {b - a:.0f} s)'); b -= 3600
            if not (0 <= a < b <= dur) or b - a > 600: raise ValueError(f'{pn} {cur}: implausible seizure {a}-{b} s of {dur} s')
            out.append((cur, float(a), float(b))); st = None
    return out


def plan_chb():
    recs = [l.strip() for l in open(fetch('chb', 'RECORDS')).read().splitlines() if l.strip().endswith('.edf')]
    with_sz = {l.strip() for l in open(fetch('chb', 'RECORDS-WITH-SEIZURES')).read().splitlines() if l.strip()}
    items, notes = [], []
    for c in sorted({int(r[3:5]) for r in recs}):
        if SMOKE and c > 3: continue
        summ = parse_summary(open(fetch('chb', f'chb{c:02d}/chb{c:02d}-summary.txt'), errors='ignore').read())
        files = [r for r in recs if r.startswith(f'chb{c:02d}/')]
        if SMOKE:
            sz = [r for r in files if summ.get(r.split('/')[1])][:2]; free = [r for r in files if r not in with_sz and not summ.get(r.split('/')[1])][:1]
            files = sz + free
        for r in files:
            s = summ.get(r.split('/')[1])
            if s is None and r in with_sz: notes.append(f'{r}: listed with seizures but absent from the summary; skipped'); continue
            if r in with_sz and not s: notes.append(f'{r}: listed with seizures but none parsed; skipped'); continue
            items.append(dict(ds='chb', rel=r, subject=f'chb{(1 if c == 21 else c):02d}', seizures=s or []))
    return items, notes


def plan_siena():
    recs = [l.strip() for l in open(fetch('siena', 'RECORDS')).read().splitlines() if l.strip().endswith('.edf')]
    pns = sorted({r.split('/')[0] for r in (SMOKE_SIENA if SMOKE else recs)})
    items, notes = [], []
    for pn in pns:
        edfs = {}
        for r in [r for r in recs if r.startswith(pn + '/')]:
            b, size = fetch_head('siena', r); H = edf_header(b); rec_len = sum(H['nsamp'])
            nrec = (size - H['hdr_bytes']) // (2 * rec_len); edfs[r.split('/')[1]] = (hms(H['start']), nrec * H['rec_dur'])
        txt = open(fetch('siena', f'{pn}/Seizures-list-{pn}.txt'), errors='ignore').read()
        sz = parse_siena(txt, pn, edfs, notes)
        for f in sorted(edfs):
            r = f'{pn}/{f}'
            if SMOKE and r not in SMOKE_SIENA: continue
            items.append(dict(ds='siena', rel=r, subject=pn, seizures=[(a, b) for g, a, b in sz if g == f]))
    return items, notes


# ================================================================== signals and features (shared with the streaming chain)
def norm_label(l, ds):
    if ds == 'chb': return l.upper().replace(' ', '')
    l = l.upper().replace('EEG', '').strip(); return SIENA_REN.get(l, l)


def open_edf(path, ds):
    """Header, channel layout and filter set-up for the 18 bipolar channels of one EDF file."""
    with open(path, 'rb') as f: raw = f.read(256); raw += f.read(256 * int(raw[252:256].decode().strip()))
    H = edf_header(raw); labs = [norm_label(l, ds) for l in H['labels']]
    need = CH18 if ds == 'chb' else SIENA_EL
    miss = [n for n in need if n not in labs]
    if miss: raise KeyError(f'missing channels {miss}')
    idx = [labs.index(n) for n in need]
    fs_in = {H['nsamp'][i] / H['rec_dur'] for i in idx}
    if len(fs_in) != 1: raise ValueError(f'mixed sampling rates {fs_in}')
    fs_in = fs_in.pop(); dec = int(round(fs_in / FS))
    if abs(fs_in - dec * FS) > 1e-6: raise ValueError(f'sampling rate {fs_in} Hz')
    pairs = None if ds == 'chb' else [(SIENA_EL.index(a), SIENA_EL.index(b)) for a, b in (c.split('-') for c in CH18)]
    rec_len = sum(H['nsamp']); nrec = (os.path.getsize(path) - H['hdr_bytes']) // (2 * rec_len)
    return dict(H=H, idx=idx, pairs=pairs, fs_in=fs_in, dec=dec, rec_len=rec_len, nrec=nrec, cum=np.cumsum([0] + H['nsamp']),
                gain=[(H['pmax'][i] - H['pmin'][i]) / (H['dmax'][i] - H['dmin'][i]) for i in idx])


def edf_block(E, blk, chans=None):
    """Data records (r, rec_len) int16 -> bipolar microvolts of the requested channels (n, samples), float64; only the
    signals these channels need are decoded."""
    H, chans, cache = E['H'], range(N_CH) if chans is None else chans, {}

    def sig(j):
        if j not in cache:
            i = E['idx'][j]; cache[j] = (blk[:, E['cum'][i]:E['cum'][i + 1]].reshape(-1).astype(np.float64) - H['dmin'][i]) * E['gain'][j] + H['pmin'][i]
        return cache[j]
    if E['pairs'] is None: return np.stack([sig(c) for c in chans])
    return np.stack([sig(E['pairs'][c][0]) - sig(E['pairs'][c][1]) for c in chans])


def make_sos(fs_in):
    from scipy.signal import butter
    return butter(4, [0.5, 45], btype='bandpass', fs=fs_in, output='sos')


def load_signals(path, ds):
    """Causal band-pass with the filter state carried across 10-min chunks, then decimation to 256 Hz."""
    from scipy.signal import sosfilt
    E = open_edf(path, ds); sos = make_sos(E['fs_in']); zi = np.zeros((sos.shape[0], N_CH, 2)); out = []
    mm = np.memmap(path, dtype='<i2', mode='r', offset=E['H']['hdr_bytes'], shape=(E['nrec'], E['rec_len']))
    step = max(1, int(round(600 / E['H']['rec_dur'])))
    for r0 in range(0, E['nrec'], step):
        y, zi = sosfilt(sos, edf_block(E, np.asarray(mm[r0:r0 + step])), axis=-1, zi=zi)
        out.append(y[:, ::E['dec']].astype(np.float32))
    del mm
    return np.concatenate(out, 1), E


FREQ = np.fft.rfftfreq(WIN, 1 / FS); BMASK = [(FREQ >= lo) & (FREQ < hi) for lo, hi in BANDS]
HANN = np.hanning(WIN).astype(np.float32)


def window_features(seg):
    """seg: (n, C, WIN) filtered EEG in microvolts -> node features (n, C, 11) float32 and upper-triangle correlations float16."""
    from scipy.stats import kurtosis
    C = seg.shape[1]; iu = np.triu_indices(C, 1)
    P = np.abs(np.fft.rfft(seg * HANN, axis=-1)) ** 2
    bp = np.stack([P[..., m].sum(-1) for m in BMASK], -1)
    ll = np.abs(np.diff(seg, axis=-1)).mean(-1); var = seg.var(-1); ku = kurtosis(seg, axis=-1)
    sb = np.signbit(seg); zc = (sb[..., 1:] != sb[..., :-1]).mean(-1)
    f = np.concatenate([np.log10(bp + 1e-6), np.log10(ll + 1e-6)[..., None], np.log10(var + 1e-6)[..., None],
                        (np.clip(ku, -3, 30) / 10)[..., None], (zc * 10)[..., None]], -1)
    z = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-6)
    Cm = np.einsum('nct,ndt->ncd', z, z, optimize=True) / seg.shape[-1]
    return np.nan_to_num(f).astype(np.float32), np.nan_to_num(Cm[:, iu[0], iu[1]]).astype(np.float16)


def file_tag(it): return it['ds'] + '__' + it['rel'].replace('/', '__').replace('+', 'p')


def process_file(it):
    """Worker: download one EDF, compute window features (non-overlapping 4-s windows, plus 1-s-hop windows inside
    seizures for training) and 128-Hz raw windows; delete the EDF."""
    tag = file_tag(it); fp = os.path.join(FILES_DIR, tag + '.npz'); rp = os.path.join(FILES_DIR, tag + '.raw.npy')
    if os.path.exists(fp) and (args.no_raw or os.path.exists(rp)): return tag, None
    try:
        path = fetch(it['ds'], it['rel']); X, E = load_signals(path, it['ds'])
    except Exception as e:
        return tag, f'{type(e).__name__}: {e}'
    T = X.shape[1]; seiz = it['seizures']
    starts = [(st, 0) for st in range(0, T - WIN + 1, WIN)]; n_main = len(starts)
    for a, e in seiz:
        starts += [(st, 1) for st in range(int(a * FS) - WIN // 2, int(e * FS) - WIN // 2, AUG_HOP) if 0 <= st <= T - WIN and st % WIN]
    Fl, Cl, Rl, M = [], [], [], []
    for b in range(0, len(starts), 512):
        chunk = starts[b:b + 512]; seg = np.stack([X[:, st:st + WIN] for st, _ in chunk])
        f, c = window_features(seg); Fl.append(f); Cl.append(c)
        if not args.no_raw: Rl.append(np.clip(seg[..., ::2], -6e4, 6e4).astype(np.float16))
        for st, aug in chunk:
            ctr = (st + WIN / 2) / FS; sid = next((j for j, (a, e) in enumerate(seiz) if a <= ctr < e), -1)
            M.append((st / FS, int(sid >= 0), aug, sid))
    info = dict(it, fs_in=E['fs_in'], hours=T / FS / 3600, n_main=n_main, edf_start=E['H']['start'])
    if not args.no_raw: np.save(rp[:-4] + '.tmp.npy', np.concatenate(Rl)); os.replace(rp[:-4] + '.tmp.npy', rp)
    np.savez(fp[:-4] + '.tmp.npz', F=np.concatenate(Fl), C=np.concatenate(Cl), M=np.array(M, np.float64), info=json.dumps(info))
    os.replace(fp[:-4] + '.tmp.npz', fp)
    if not args.keep_edf and it['rel'] != EQUIV_FILE:
        try: os.remove(path)
        except OSError: pass
    return tag, None


# ================================================================== figures only (also used at the end of the run)
def make_figures(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    FIG = os.path.join(R, 'figures'); os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
    COL = {'BRIDGE': '#1f5a96', 'BRIDGE-R': '#d0661f', 'BRIDGE-A': '#7aa6d6', 'Transformer': '#8c8c8c', 'MLP': '#b5b5b5',
           'LogReg': '#2e7d32', 'GBDT': '#8bc34a', 'GBDT-R': '#c0ca33', 'RawCNN': '#6d4c41'}

    def rd(name):
        p = os.path.join(R, name)
        if not os.path.exists(p): return []
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            for k, v in r.items():
                try: r[k] = float(v)
                except (TypeError, ValueError): pass
        return rows

    def save(fig, name):
        fig.savefig(os.path.join(FIG, name + '.png'), dpi=600, bbox_inches='tight'); fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
        plt.close(fig); print('figure', name, flush=True)
    runs = [r for r in rd('results_runs.csv') if r.get('seed') == 0]
    get = lambda e, m, n, op='f1': next((r for r in runs if r['exp'] == e and r['model'] == m and r['nch'] == n and r['op'] == op), None)
    ds = json.load(open(os.path.join(R, 'dataset.json'))) if os.path.exists(os.path.join(R, 'dataset.json')) else None

    if ds:   # Fig. 2: the two hospitals
        fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw=dict(width_ratios=[1.7, 1.2, 0.8], wspace=0.55))
        for k, (d, col) in enumerate((('CHB-MIT', '#1f5a96'), ('Siena', '#d0661f'))):
            S = [s for s in ds['subjects'] if s['dataset'] == d]
            x = np.arange(len(S)); axs[k].bar(x, [s['hours'] for s in S], color=col, alpha=0.35)
            ax2 = axs[k].twinx(); ax2.plot(x, [s['seizures'] for s in S], 'o', color=col, ms=3); ax2.spines['top'].set_visible(False)
            axs[k].set_xticks(x); axs[k].set_xticklabels([s['name'].replace('chb', '') for s in S], rotation=90, fontsize=5.5)
            axs[k].set_ylabel('Recording (h), bars'); ax2.set_ylabel('Seizures, dots'); axs[k].set_title(f'({"ab"[k]}) {d}', loc='left')
        ages = [(d, [s['age'] for s in ds['subjects'] if s['dataset'] == d and s.get('age') is not None]) for d in ('CHB-MIT', 'Siena')]
        axs[2].boxplot([a for _, a in ages], widths=0.5); axs[2].set_xticks([1, 2]); axs[2].set_xticklabels([d for d, _ in ages])
        axs[2].set_ylabel('Age (years)'); axs[2].set_title('(c) age', loc='left'); fig.subplots_adjust(left=0.07, right=0.985, bottom=0.2, top=0.9, wspace=0.8); save(fig, 'fig2_hospitals')

    if runs:   # Fig. 3: in-hospital against cross-hospital
        ms = [m for m in M18 if get('x_c2s', m, 18)]
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
        for k, (ein, ex, tit) in enumerate((('in_siena', 'x_c2s', '(a) test hospital: Siena'), ('in_chb', 'x_s2c', '(b) test hospital: CHB-MIT'))):
            x = np.arange(len(ms))
            for j, (e, hatch, lab) in enumerate(((ein, None, 'trained in the same hospital'), (ex, '///', 'trained in the other hospital'))):
                v = [(get(e, m, 18) or {}).get('pooled_f1', np.nan) for m in ms]
                axs[k].bar(x + (j - 0.5) * 0.38, v, 0.38, color=[COL[m] for m in ms], hatch=hatch, edgecolor='white' if hatch else None,
                           alpha=1 if j == 0 else 0.75, label=lab)
            axs[k].set_xticks(x); axs[k].set_xticklabels(ms, rotation=35, ha='right'); axs[k].set_ylabel('Event F1 (SzCORE)')
            axs[k].set_title(tit, loc='left'); axs[k].set_ylim(0, 1)
        axs[0].legend(handles=[Patch(facecolor='#9e9e9e', label='trained in the same hospital'),
                               Patch(facecolor='#9e9e9e', alpha=0.75, hatch='///', edgecolor='white', label='trained in the other hospital')],
                      loc='upper right', frameon=False); fig.tight_layout(); save(fig, 'fig3_in_vs_cross')

    sw = rd('sweeps.csv')
    if sw:     # Fig. 4: sensitivity against false alarms per 24 h
        fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=True)
        for k, e in enumerate(EXPS):
            for m in ('BRIDGE', 'BRIDGE-R', 'BRIDGE-A', 'GBDT', 'GBDT-R', 'RawCNN', 'LogReg'):
                rr = sorted([r for r in sw if r['exp'] == e and r['model'] == m and r['nch'] == 18 and r['seed'] == 0], key=lambda r: r['fp24'])
                if rr: axs[k].plot([max(r['fp24'], 0.1) for r in rr], [r['sens'] for r in rr], color=COL[m], lw=1.0, label=m)
            r0 = get(e, 'BRIDGE-R', 18)
            if r0: axs[k].plot(max(r0['pooled_fp24'], 0.1), r0['pooled_sens'], 'k*', ms=5)
            axs[k].set_xscale('log'); axs[k].set_xlim(0.1, 1000); axs[k].set_xlabel('False alarms / 24 h'); axs[k].set_title(f'({"abcd"[k]}) {EXP_LABEL[e]}', loc='left', fontsize=7)
        axs[0].set_ylabel('Event sensitivity'); axs[0].set_ylim(0, 1); h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=6.5, ncol=7, loc='lower center', bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig4_sens_fa_curves')

    if runs:   # Fig. 6: fewer channels
        fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
        for k, e in enumerate(EXPS):
            for m in MSUB:
                v = [(get(e, m, n) or {}).get('pooled_f1', np.nan) for n in (18, 4, 2)]
                axs[k].plot([0, 1, 2], v, 'o-', color=COL[m], lw=1, ms=3, label=m)
            axs[k].set_xticks([0, 1, 2]); axs[k].set_xticklabels(['18', '4', '2']); axs[k].set_xlabel('Channels')
            axs[k].set_title(f'({"abcd"[k]}) {EXP_LABEL[e]}', loc='left', fontsize=7)
        axs[0].set_ylabel('Event F1'); axs[0].set_ylim(0, 1); axs[-1].legend(frameon=False, fontsize=5.5); fig.tight_layout(); save(fig, 'fig6_channels')

    sh = rd('shift.csv')
    if sh:     # Fig. 7: hospital identifiability
        views = ['absolute', 'referenced', 'both']
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw=dict(width_ratios=[1, 2.2]))
        v = [next((r['auroc'] for r in sh if r['kind'] == 'hospital_auroc' and r['view'] == w), np.nan) for w in views]
        axs[0].bar(range(3), v, color=['#7aa6d6', '#d0661f', '#1f5a96'], width=0.6)
        for i, x in enumerate(v): axs[0].text(i, x + 0.01, f'{x:.2f}', ha='center', fontsize=7)
        axs[0].axhline(0.5, color='k', lw=0.6, ls=':'); axs[0].set_xticks(range(3)); axs[0].set_xticklabels(views)
        axs[0].set_ylim(0.4, 1.05); axs[0].set_ylabel('AUROC, hospital from EEG'); axs[0].set_title('(a) which hospital?', loc='left')
        fn = [r for r in sh if r['kind'] == 'feature_d']; names = list(dict.fromkeys(r['feature'] for r in fn)); x = np.arange(len(names))
        for j, (w, col) in enumerate((('absolute', '#7aa6d6'), ('referenced', '#d0661f'))):
            axs[1].bar(x + (j - 0.5) * 0.38, [next((r['d_hospital'] for r in fn if r['feature'] == n and r['view'] == w), np.nan) for n in names], 0.38, color=col, label=w)
        axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=35, ha='right'); axs[1].set_ylabel('|Cohen d|, CHB-MIT vs Siena')
        axs[1].legend(frameon=False); axs[1].set_title('(b) hospital difference per feature (interictal)', loc='left'); fig.tight_layout(); save(fig, 'fig7_hospital_shift')

    if runs:   # Fig. 5: calibrated against oracle threshold
        ms = [m for m in M18 if get('x_c2s', m, 18)]
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
        for k, e in enumerate(('x_c2s', 'x_s2c')):
            x = np.arange(len(ms))
            for j, (op, lab, a) in enumerate((('f1', 'threshold from the training hospital', 1), ('oracle', 'best threshold on the test hospital (oracle)', 0.45))):
                axs[k].bar(x + (j - 0.5) * 0.38, [(get(e, m, 18, op) or {}).get('pooled_f1', np.nan) for m in ms], 0.38, color=[COL[m] for m in ms], alpha=a, label=lab)
            axs[k].set_xticks(x); axs[k].set_xticklabels(ms, rotation=35, ha='right'); axs[k].set_ylabel('Event F1'); axs[k].set_ylim(0, 1)
            axs[k].set_title(f'({"ab"[k]}) {EXP_LABEL[e]}', loc='left')
        axs[0].legend(handles=[Patch(facecolor='#616161', label='threshold from the training hospital'),
                               Patch(facecolor='#616161', alpha=0.45, label='best threshold on the test hospital (oracle)')],
                      frameon=False, fontsize=6, loc='upper left'); fig.tight_layout(); save(fig, 'fig5_threshold_transfer')

    sub = rd('results_subjects.csv')
    if sub:    # Fig. 8: sensitivity per subject in and across hospitals
        fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.6))
        age = {x['name']: x.get('age') for x in ds['subjects']} if ds else {}
        for k, (ein, ex, lin, lx, tit) in enumerate((('in_siena', 'x_c2s', 'trained on Siena', 'trained on CHB-MIT', '(a) test hospital: Siena (adults)'),
                                                     ('in_chb', 'x_s2c', 'trained on CHB-MIT', 'trained on Siena', '(b) test hospital: CHB-MIT (children)'))):
            S = sorted({r['subject'] for r in sub if r['exp'] == ein}); x = np.arange(len(S))
            for j, (e, lab, col) in enumerate(((ein, lin, '#d0661f'), (ex, lx, '#1f5a96'))):
                v = [next((r['sens'] for r in sub if r['exp'] == e and r['model'] == 'BRIDGE-R' and r['nch'] == 18 and r['subject'] == s and r['seed'] == 0 and r['op'] == 'f1'), np.nan) for s in S]
                axs[k].bar(x + (j - 0.5) * 0.38, v, 0.38, color=col, label='BRIDGE-R, ' + lab)
            lab = [f"{s.replace('chb', '')}\n{age[s]:g} y" if age.get(s) is not None else s.replace('chb', '') for s in S]
            axs[k].set_xticks(x); axs[k].set_xticklabels(lab, fontsize=5.5); axs[k].set_ylabel('Event sensitivity'); axs[k].set_ylim(0, 1.05)
            axs[k].set_title(tit, loc='left'); axs[k].legend(frameon=False, ncol=2, fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 1.25))
        fig.tight_layout(); save(fig, 'fig8_subjects')

    st = rd('stream_timing.csv')
    if st:     # Fig. 10: streaming chain cost
        stages = ['filter', 'features', 'reference', 'model', 'alarm']
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
        cfgs = [r for r in st if r['model'] == 'BRIDGE-R']; lab = [f"{int(r['nch'])} ch, {r['precision']}" for r in cfgs]; left = np.zeros(len(cfgs))
        cols = ['#bdbdbd', '#1f5a96', '#d0661f', '#2e7d32', '#6d4c41']
        for s, c in zip(stages, cols):
            v = np.array([r[f'ms_{s}'] for r in cfgs]); axs[0].barh(range(len(cfgs)), v, left=left, color=c, label=s); left += v
        axs[0].set_yticks(range(len(cfgs))); axs[0].set_yticklabels(lab); axs[0].invert_yaxis(); axs[0].set_xlabel('CPU time per 4-s window (ms, one thread)')
        axs[0].legend(frameon=False, fontsize=6, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.28)); axs[0].set_title('(a) time per stage, BRIDGE-R', loc='left')
        mem = rd('stream_memory.csv')
        if mem:
            mm = [r for r in mem if r['model'] == 'BRIDGE-R' and r['precision'] == 'fp32']
            axs[1].bar(range(len(mm)), [r['state_kb'] for r in mm], color='#bdbdbd', label='signal and reference state')
            axs[1].bar(range(len(mm)), [r['param_kb'] for r in mm], bottom=[r['state_kb'] for r in mm], color='#1f5a96', label='parameters')
            axs[1].set_xticks(range(len(mm))); axs[1].set_xticklabels([f"{int(r['nch'])} channels" for r in mm]); axs[1].set_ylabel('Memory (kB, fp32)')
            axs[1].legend(frameon=False, fontsize=6); axs[1].set_title('(b) memory of the streaming detector', loc='left')
        fig.tight_layout(); save(fig, 'fig10_streaming_cost')

    tl = rd('timeline.csv')
    if tl:     # Fig. 9: one Siena seizure
        fig, ax = plt.subplots(figsize=(7.0, 1.9)); t = np.array([r['t'] for r in tl]) / 60
        for c, col, lab in (('p_in', '#d0661f', 'BRIDGE-R, trained on Siena'), ('p_cross', '#1f5a96', 'BRIDGE-R, trained on CHB-MIT')):
            ax.plot(t, [r[c] for r in tl], color=col, lw=0.8, label=lab)
        on, off = tl[0]['onset'] / 60, tl[0]['offset'] / 60; ax.axvspan(on, off, color='#e53935', alpha=0.2, lw=0, label='seizure (annotated)')
        ax.set_xlabel('Time in recording (min)'); ax.set_ylabel('Seizure probability'); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, ncol=3, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 1.22)); ax.set_title(tl[0]['file'], loc='left', fontsize=7)
        fig.tight_layout(); save(fig, 'fig9_timeline')


if args.figs:
    LOG = None; make_figures(args.figs); sys.exit(0)

os.makedirs(args.out, exist_ok=True); os.makedirs(args.data, exist_ok=True)
LOG = open(os.path.join(args.out, 'log.txt'), 'a')
FILES_DIR = os.path.join(args.data, 'files' + ('_smoke' if SMOKE else '')); os.makedirs(FILES_DIR, exist_ok=True)
for d in ('ck', 'preds', 'models'): os.makedirs(os.path.join(args.out, d), exist_ok=True)
warnings.filterwarnings('ignore', message='.*quantiz.*'); warnings.filterwarnings('ignore', category=UserWarning)
log(f'start; smoke={SMOKE}; python {sys.version.split()[0]}')

# ================================================================== stage prep: plan, download, features, assembly
ASM = os.path.join(args.data, f'assembled_v{V}' + ('_smoke' if SMOKE else '') + '.npz')
RAWF = os.path.join(args.data, f'raw128_v{V}' + ('_smoke' if SMOKE else '') + '.f16')
if not (os.path.exists(ASM) and (args.no_raw or os.path.exists(RAWF))):
    plan_c, notes_c = plan_chb(); plan_s, notes_s = plan_siena(); PLAN = plan_c + plan_s
    log(f'planned {len(plan_c)} CHB-MIT and {len(plan_s)} Siena recordings; {len(notes_c) + len(notes_s)} annotation notes')
    for n in notes_c + notes_s: log('  note:', n)
    todo = [it for it in PLAN if not os.path.exists(os.path.join(FILES_DIR, file_tag(it) + '.npz'))]
    try: avail = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1e9
    except (ValueError, OSError): avail = 8.0
    W = args.workers or max(1, min(8, os.cpu_count() or 2, int(avail // 3)))
    log(f'{len(todo)} recordings to download and process with {W} workers ({avail:.0f} GB RAM free)')
    fails = {}
    if todo:
        t1 = time.time(); done = 0
        with cf.ProcessPoolExecutor(W, mp_context=mp.get_context('fork')) as ex:
            for tag, err in ex.map(process_file, todo, chunksize=1):
                done += 1
                if err: fails[tag] = err; log('  skipped', tag, err)
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t1; log(f'  {done}/{len(todo)} recordings processed; about {el / done * (len(todo) - done) / 60:.0f} min left')
    ok = [it for it in PLAN if os.path.exists(os.path.join(FILES_DIR, file_tag(it) + '.npz'))]
    log(f'assembling {len(ok)} recordings')
    sizes = [int(np.load(os.path.join(FILES_DIR, file_tag(it) + '.npz'))['M'].shape[0]) for it in ok]; N = sum(sizes)
    subj_names = sorted({it['subject'] for it in ok if it['ds'] == 'chb'}) + sorted({it['subject'] for it in ok if it['ds'] == 'siena'})
    Fl, Cl, Ml, FILES, SZ = [], [], [], [], []
    RM = None if args.no_raw else np.memmap(RAWF + '.tmp', dtype=np.float16, mode='w+', shape=(max(N, 1), N_CH, WIN // 2))
    k = 0
    for fid, it in enumerate(ok):
        d = np.load(os.path.join(FILES_DIR, file_tag(it) + '.npz')); info = json.loads(str(d['info'])); M = d['M']; n = len(M)
        sid0 = len(SZ)
        for j, (a, e) in enumerate(info['seizures']): SZ.append(dict(fid=fid, onset=a, offset=e))
        sid = np.where(M[:, 3] >= 0, M[:, 3] + sid0, -1)
        Ml.append(np.column_stack([np.full(n, DS_ID[it['ds']]), np.full(n, subj_names.index(it['subject'])), np.full(n, fid), M[:, 0], M[:, 1], M[:, 2], sid]))
        Fl.append(d['F']); Cl.append(d['C'])
        if RM is not None: RM[k:k + n] = np.load(os.path.join(FILES_DIR, file_tag(it) + '.raw.npy'))
        k += n; FILES.append(dict(ds=it['ds'], rel=it['rel'], subject=it['subject'], hours=info['hours'], n_main=info['n_main'],
                                  seizures=info['seizures'], fs_in=info['fs_in'], edf_start=info['edf_start']))
    if RM is not None: RM.flush(); del RM; os.replace(RAWF + '.tmp', RAWF)
    np.savez(ASM + '.tmp.npz', F=np.concatenate(Fl), C=np.concatenate(Cl), META=np.concatenate(Ml), files=json.dumps(FILES),
             seizures=json.dumps(SZ), subjects=json.dumps(subj_names), notes=json.dumps(notes_c + notes_s), fails=json.dumps(fails))
    os.replace(ASM + '.tmp.npz', ASM); del Fl, Cl, Ml
    if not args.no_raw:
        for it in ok:
            p = os.path.join(FILES_DIR, file_tag(it) + '.raw.npy')
            if os.path.exists(p): os.remove(p)
    log(f'assembled {N} windows')

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

D = np.load(ASM, allow_pickle=False)
Fa, Ca, META = D['F'], D['C'], D['META']
FILES, SZ, SUBJ_NAMES = json.loads(str(D['files'])), json.loads(str(D['seizures'])), json.loads(str(D['subjects']))
NOTES, FAILS = json.loads(str(D['notes'])), json.loads(str(D['fails']))
DSID, SUBJ, FID = META[:, 0].astype(int), META[:, 1].astype(int), META[:, 2].astype(int)
T0, Y, AUG, SID = META[:, 3], META[:, 4].astype(int), META[:, 5].astype(bool), META[:, 6].astype(int)
N = len(Y)
SUBJ_DS = {s: DS_ID['chb'] if n.startswith('chb') else DS_ID['siena'] for s, n in enumerate(SUBJ_NAMES)}
FILE_WIN = {}
for f in range(len(FILES)):
    ii = np.where((FID == f) & ~AUG)[0]; FILE_WIN[f] = ii[np.argsort(T0[ii])]
FILES_OF = {s: [f for f, fi in enumerate(FILES) if SUBJ_NAMES.index(fi['subject']) == s] for s in range(len(SUBJ_NAMES))}

# dataset description
try:
    chb_age = {}
    for line in open(fetch('chb', 'SUBJECT-INFO')).read().splitlines()[1:]:
        p = line.split()
        if len(p) >= 3 and p[0].startswith('chb'): chb_age[p[0][:5]] = float(p[2])
    sie_age = {}
    for r in csv.DictReader(open(fetch('siena', 'subject_info.csv')), skipinitialspace=True): sie_age[r['patient_id'].strip()] = float(r['age_years'])
except Exception as e:
    chb_age, sie_age = {}, {}; log('age table not read:', e)
DATASET = dict(subjects=[], notes=NOTES, skipped=FAILS, windows=int((~AUG).sum()), ictal_windows=int(((Y == 1) & ~AUG).sum()), augmented=int(AUG.sum()))
for s, n in enumerate(SUBJ_NAMES):
    fl = FILES_OF[s]
    DATASET['subjects'].append(dict(name=n, dataset=DS_NAME[SUBJ_DS[s]], files=len(fl), hours=sum(FILES[f]['hours'] for f in fl),
                                    seizures=sum(len(FILES[f]['seizures']) for f in fl), age=(chb_age if n.startswith('chb') else sie_age).get(n)))
for d in (0, 1):
    S = [x for x in DATASET['subjects'] if x['dataset'] == DS_NAME[d]]
    DATASET[DS_NAME[d]] = dict(subjects=len(S), files=sum(x['files'] for x in S), hours=sum(x['hours'] for x in S), seizures=sum(x['seizures'] for x in S))
    log(f"{DS_NAME[d]}: {len(S)} subjects, {DATASET[DS_NAME[d]]['files']} recordings, {DATASET[DS_NAME[d]]['hours']:.1f} h, {DATASET[DS_NAME[d]]['seizures']} seizures")
json.dump(DATASET, open(os.path.join(args.out, 'dataset.json'), 'w'), indent=1)
if args.stop_after == 'prep': sys.exit(0)


# ================================================================== causal patient reference
def patient_reference():
    B = np.empty((N, N_CH, N_LOG), np.float32)
    for f in range(len(FILES)):
        ii = np.where(FID == f)[0]; na = FILE_WIN[f]
        if not len(na): continue
        med = pd.DataFrame(Fa[na, :, :N_LOG].reshape(len(na), -1)).rolling(W_BASE, min_periods=1).median().to_numpy(np.float32).reshape(-1, N_CH, N_LOG)
        B[ii] = med[np.clip(np.searchsorted(T0[na], T0[ii] + 1e-6, side='right') - 1, 0, len(na) - 1)]
    return B


t1 = time.time(); FR = np.concatenate([Fa, Fa[..., :N_LOG] - patient_reference()], -1); del Fa
log(f'patient reference in {time.time() - t1:.0f} s; {N} windows ({(~AUG).sum()} for evaluation)')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
Ft = torch.tensor(FR, device=dev); Ct = torch.tensor(Ca, device=dev); Yt = torch.tensor(Y, device=dev)
Rt = None
if not args.no_raw and os.path.exists(RAWF):
    rb = N * N_CH * (WIN // 2) * 2; RM = np.memmap(RAWF, dtype=np.float16, mode='r', shape=(N, N_CH, WIN // 2))
    try: avail = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (ValueError, OSError): avail = 0
    on_gpu = dev.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory > rb + 14e9
    if on_gpu or avail > 1.4 * rb:
        Rt = torch.empty((N, N_CH, WIN // 2), dtype=torch.float16, device=dev if on_gpu else 'cpu')
        for b in range(0, N, 50000): Rt[b:b + 50000] = torch.from_numpy(np.asarray(RM[b:b + 50000])).to(Rt.device)
    else: Rt = RM
    log(f'raw signals: {rb / 1e9:.1f} GB on {"GPU" if on_gpu else ("CPU memory" if isinstance(Rt, torch.Tensor) else "disk")}')
log(f'device {dev}; torch {torch.__version__}')


# ================================================================== models
POS = {'FP1': (-0.31, 0.95), 'FP2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.51), 'FZ': (0, 0.47),
       'F4': (0.41, 0.51), 'F8': (0.81, 0.59), 'T7': (-1, 0), 'C3': (-0.5, 0), 'CZ': (0, 0), 'C4': (0.5, 0), 'T8': (1, 0),
       'P7': (-0.81, -0.59), 'P3': (-0.41, -0.51), 'PZ': (0, -0.47), 'P4': (0.41, -0.51), 'P8': (0.81, -0.59),
       'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}


def static_graph(chs):
    """Channels sharing an electrode: 1; others: Gaussian of the distance between bipolar midpoints (< 0.6); row-normalized."""
    names = [CH18[i] for i in chs]; mid = np.array([np.mean([POS[e] for e in c.split('-')], 0) for c in names]); n = len(chs)
    A = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = np.linalg.norm(mid[i] - mid[j])
            A[i, j] = 1.0 if set(names[i].split('-')) & set(names[j].split('-')) else (math.exp(-(d / 0.35) ** 2) if d < 0.6 else 0.0)
    s = A.sum(1, keepdims=True); return np.where(s > 0, A / np.maximum(s, 1e-9), 0).astype(np.float32)


class Sub:
    """A channel subset: graph, correlation pairs and indices."""
    def __init__(s, chs):
        s.chs = sorted(chs); s.n = len(s.chs); s.A = static_graph(s.chs); s.iu = np.triu_indices(s.n, 1); s.k = max(1, min(4, s.n - 1))
        iu18 = np.triu_indices(N_CH, 1); pos = {(a, b): k for k, (a, b) in enumerate(zip(*iu18))}
        s.pairs = np.array([pos[(s.chs[i], s.chs[j])] for i, j in zip(*s.iu)], np.int64)
        s.ch_t = torch.tensor(s.chs, device=dev); s.pairs_t = torch.tensor(s.pairs, device=dev)


SUBS = {n: Sub(c) for n, c in SUBSETS.items()}


def topk_norm(M, k):
    M = M.abs(); v, i = M.topk(k, dim=-1); A = torch.zeros_like(M).scatter_(-1, i, v); A = 0.5 * (A + A.transpose(1, 2))
    return A / (A.sum(-1, keepdim=True) + 1e-6)


class GLayer(nn.Module):
    def __init__(s, d):
        super().__init__(); s.a = nn.Linear(d, d); s.b = nn.Linear(d, d, bias=False); s.n = nn.LayerNorm(d); s.dp = nn.Dropout(0.1)

    def forward(s, h, A): return s.n(h + s.dp(F_.gelu(s.a(h) + torch.matmul(A, s.b(h)))))


class BRIDGE(nn.Module):
    """Dual-graph encoder of one 4-s window (anatomical graph and the window's top-k correlation graph, mixed with learned
    weights; 2 message-passing layers; mean and max pooling) and a small classifier."""
    def __init__(s, f, sub, d=32):
        super().__init__(); s.n, s.k = sub.n, sub.k; s.inp = nn.Linear(f, d); s.L = nn.ModuleList([GLayer(d), GLayer(d)]); s.mix = nn.Parameter(torch.zeros(2))
        s.register_buffer('As', torch.tensor(sub.A)); s.register_buffer('iu', torch.tensor(np.stack(sub.iu)))
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 2))

    def forward(s, x, c):
        M = torch.zeros(c.shape[0], s.n, s.n, device=c.device, dtype=torch.float32); M[:, s.iu[0], s.iu[1]] = c.float(); M = M + M.transpose(1, 2)
        w = torch.softmax(s.mix, 0); A = w[0] * s.As + w[1] * topk_norm(M, s.k); h = F_.gelu(s.inp(x))
        for layer in s.L: h = layer(h, A)
        return s.head(torch.cat([h.mean(1), h.amax(1)], -1))


class ChanTransformer(nn.Module):
    def __init__(s, f, sub, d=32):
        super().__init__(); s.inp = nn.Linear(f, d); s.pos = nn.Parameter(torch.zeros(1, sub.n, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 2))

    def forward(s, x, c): h = s.enc(s.inp(x) + s.pos); return s.head(torch.cat([h.mean(1), h.amax(1)], -1))


class MLP(nn.Module):
    def __init__(s, f, sub):
        super().__init__(); s.net = nn.Sequential(nn.Flatten(), nn.Linear(sub.n * f, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(s, x, c): return s.net(x)


class RawCNN(nn.Module):
    def __init__(s, sub):
        super().__init__(); L, ch = [], [sub.n, 32, 64, 64, 64]
        for a, b in zip(ch[:-1], ch[1:]): L += [nn.Conv1d(a, b, 7, 2, 3), nn.BatchNorm1d(b), nn.ReLU()]
        s.net = nn.Sequential(*L, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(s, r, c=None): r = r.float(); return s.net((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-3))


N_IN = {'ref': N_F + N_LOG, 'rel': N_F, 'abs': N_F}
GBDT_NOTE = []


def make_net(arch, kind, sub):
    f = N_IN.get(kind)
    return {'gnn': lambda: BRIDGE(f, sub), 'tf': lambda: ChanTransformer(f, sub), 'mlp': lambda: MLP(f, sub), 'cnn': lambda: RawCNN(sub)}[arch]().to(dev)


def norm_stats(idx):
    sel = idx[~AUG[idx]]
    if len(sel) > 200000: sel = np.random.default_rng(0).choice(sel, 200000, replace=False)
    x = Ft[torch.as_tensor(sel, device=dev)]; return x.mean(0), x.std(0) + 1e-4


def feats(bt, kind, st, sub):
    x = (Ft[bt][:, sub.ch_t] - st[0][sub.ch_t]) / st[1][sub.ch_t]
    if kind == 'rel': x = torch.cat([x[..., N_F:], x[..., N_LOG:N_F]], -1)
    elif kind == 'abs': x = x[..., :N_F]
    return x, Ct[bt][:, sub.pairs_t]


def raw_batch(bi, sub):
    if isinstance(Rt, torch.Tensor):
        return Rt[torch.as_tensor(bi, device=Rt.device)][:, sub.ch_t.to(Rt.device)].to(dev, non_blocking=True)
    o = np.sort(bi); back = np.argsort(np.argsort(bi))
    return torch.from_numpy(np.asarray(Rt[o])[back][:, sub.chs]).to(dev)


def balanced(tr, rng):
    pos, neg = tr[Y[tr] == 1], tr[Y[tr] == 0]
    return np.concatenate([pos, rng.choice(neg, min(len(neg), RATIO * max(len(pos), 1)), replace=False)])


def fit(model, nch, tr, seed):
    arch, kind = MODELS[model]; sub = SUBS[nch]; rng = np.random.default_rng(seed); torch.manual_seed(seed)
    st = norm_stats(tr) if kind != 'raw' else None
    if arch in ('lin', 'gbdt'):
        idx = balanced(tr, rng); x, _ = feats(torch.as_tensor(idx, device=dev), kind, st, sub); x = x.reshape(len(idx), -1); y = Yt[torch.as_tensor(idx, device=dev)]
        if arch == 'lin':   # L2 logistic regression (C = 0.1), full-batch L-BFGS
            w = torch.zeros(x.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
            opt = torch.optim.LBFGS([w, b], lr=1, max_iter=200, line_search_fn='strong_wolfe'); lam = 1.0 / (2 * 0.1 * len(idx))

            def closure():
                opt.zero_grad(); loss = F_.binary_cross_entropy_with_logits(x @ w + b, y.float()) + lam * (w * w).sum(); loss.backward(); return loss
            opt.step(closure); return dict(arch=arch, kind=kind, sub=sub, st=st, w=w.detach(), b=b.detach())
        xn, yn = x.cpu().numpy(), y.cpu().numpy()
        try:
            import xgboost as xgb
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                                    tree_method='hist', device='cuda' if dev.type == 'cuda' else 'cpu', random_state=seed, n_jobs=-1, verbosity=0)
            clf.fit(xn, yn)
        except Exception as e:
            if not GBDT_NOTE: GBDT_NOTE.append(str(e)); log('xgboost unavailable; scikit-learn histogram gradient boosting used:', e)
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, l2_regularization=1.0, random_state=seed).fit(xn, yn)
        return dict(arch=arch, kind=kind, sub=sub, st=st, clf=clf)
    m = make_net(arch, kind, sub); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    for ep in range(EPOCHS_RAW if arch == 'cnn' else EPOCHS):
        idx = balanced(tr, rng); rng.shuffle(idx); m.train()
        for b in range(0, len(idx), BS):
            bi = idx[b:b + BS]; bt = torch.as_tensor(bi, device=dev)
            if arch == 'cnn': out = m(raw_batch(bi, sub))
            else: x, c = feats(bt, kind, st, sub); out = m(x, c)
            loss = F_.cross_entropy(out, Yt[bt]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval(); return dict(arch=arch, kind=kind, sub=sub, st=st, m=m)


@torch.no_grad()
def predict(fm, idx):
    out = []; arch = fm['arch']; B = 50000 if arch in ('lin', 'gbdt') else (4096 if arch == 'cnn' else 16384)
    for b in range(0, len(idx), B):
        bi = idx[b:b + B]; bt = torch.as_tensor(bi, device=dev)
        if arch == 'cnn': p = torch.softmax(fm['m'](raw_batch(bi, fm['sub'])).float(), -1)[:, 1]
        else:
            x, c = feats(bt, fm['kind'], fm['st'], fm['sub'])
            if arch == 'lin': p = torch.sigmoid(x.reshape(len(bi), -1) @ fm['w'] + fm['b'])
            elif arch == 'gbdt': p = torch.tensor(fm['clf'].predict_proba(x.reshape(len(bi), -1).cpu().numpy())[:, 1])
            else: p = torch.softmax(fm['m'](x, c).float(), -1)[:, 1]
        out.append(p.float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


# ================================================================== SzCORE event scoring
def merge_split(ev):
    out = []
    for s, e in ev:
        if out and s - out[-1][1] < MERGE_GAP: out[-1] = (out[-1][0], e)
        else: out.append((s, e))
    res = []
    for s, e in out:
        while e - s > MAX_EVENT: res.append((s, s + MAX_EVENT)); s += MAX_EVENT
        res.append((s, e))
    return res


REF_EV = {f: merge_split(sorted((z['onset'], z['offset']) for z in SZ if z['fid'] == f)) for f in range(len(FILES))}
DUR = {f: len(FILE_WIN[f]) * WIN_S for f in range(len(FILES))}


def alarm_events(p, t, t0):
    pos = (p >= t).astype(np.int8)
    if not pos.any(): return []
    al = np.convolve(pos, np.ones(N_ALARM, np.int8))[:len(p)] >= K_ALARM
    if not al.any(): return []
    d = np.diff(np.r_[0, al.astype(np.int8), 0]); i0, i1 = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return merge_split([(float(t0[a]), float(t0[b]) + WIN_S) for a, b in zip(i0, i1)])


def file_counts(f, p, t):
    """(TP, FP, reference events, latencies) of one recording at threshold t (SzCORE event scoring)."""
    ref = REF_EV[f]; hyp = alarm_events(p, t, T0[FILE_WIN[f]])
    if not ref: return 0, len(hyp), 0, []
    if not hyp: return 0, 0, len(ref), []
    L = int(math.ceil(DUR[f])); hm = np.zeros(L + 1, bool); tpm = np.zeros(L + 1, bool)
    for s, e in hyp: hm[int(round(s)):int(round(e))] = True
    tp, lat = 0, []
    for on, off in ref:
        a, b = int(round(max(0, on - TOL_PRE))), int(round(min(L, off + TOL_POST))); seg = hm[a:b]
        if seg.any(): tp += 1; tpm[a:b] = True; lat.append(a + int(np.argmax(seg)) - on)
    fp = sum(1 for s, e in hyp if not tpm[int(round(s)):int(round(e))].any())   # false alarm: entirely outside detected seizures
    return tp, fp, len(ref), lat


def counts(P, files, t):
    tp = fp = nr = 0; dur = 0.0; lat = []
    for f in files:
        a, b, c, l = file_counts(f, P[FILE_WIN[f]], t); tp += a; fp += b; nr += c; dur += DUR[f]; lat += l
    return tp, fp, nr, dur, lat


def scores(tp, fp, nr, dur):
    """Sensitivity, precision, F1 and false alarms per 24 h as defined by SzCORE (timescoring)."""
    return dict(sens=tp / nr if nr else np.nan, prec=tp / (tp + fp) if tp + fp else np.nan,
                f1=2 * tp / (2 * tp + fp + nr - tp) if nr + fp else np.nan, fp24=fp / max(dur / 86400, 1e-9))


def sweep(P, files, grid):
    rows = []
    for t in grid:
        tp, fp, nr, dur, _ = counts(P, files, t); rows.append(dict(t=float(t), tp=tp, fp=fp, nref=nr, **scores(tp, fp, nr, dur)))
    return rows


def calibrate(P, files):
    pv = P[np.concatenate([FILE_WIN[f] for f in files])]
    grid = np.unique(np.r_[THR_GRID, np.quantile(pv, np.clip(1 - np.logspace(-4.5, -0.7, 30), 0, 1))])
    sw = sweep(P, files, grid); f1 = np.array([r['f1'] if np.isfinite(r['f1']) else -1 for r in sw])
    best = int(np.where(f1 == f1.max())[0][-1]); out = dict(f1=sw[best]['t'], inner_f1=float(f1.max()))
    for b in BUDGETS:
        ok = [r['t'] for r in sw if r['fp24'] <= b]; out[f'fa{b}'] = float(min(ok)) if ok else float(grid[-1])
    return out


# ================================================================== experiments
def groups(subs, k, seed):
    p = np.random.default_rng(seed).permutation(subs); return [sorted(int(x) for x in g) for g in np.array_split(p, min(k, len(subs)))]


CHB_S = sorted(s for s in range(len(SUBJ_NAMES)) if SUBJ_DS[s] == 0); SIE_S = sorted(s for s in range(len(SUBJ_NAMES)) if SUBJ_DS[s] == 1)
FOLDS = {'x_c2s': [(CHB_S, SIE_S, groups(CHB_S, 3, 1))], 'x_s2c': [(SIE_S, CHB_S, groups(SIE_S, 3, 1))],
         'in_siena': [([s for s in SIE_S if s != t], [t], groups([s for s in SIE_S if s != t], 2, 1)) for t in SIE_S],
         'in_chb': [([s for s in CHB_S if s not in g], g, groups([s for s in CHB_S if s not in g], 2, 1)) for g in groups(CHB_S, 4, 0)]}
EVAL = ~AUG
win_of = lambda subs, ev=False: np.where(np.isin(SUBJ, subs) & (EVAL if ev else True))[0]
files_of = lambda subs: [f for s in subs for f in FILES_OF[s]]
RES_P = os.path.join(args.out, 'results.pkl')
RES = pickle.load(open(RES_P, 'rb')) if os.path.exists(RES_P) else dict(runs=[], subjects=[], sweeps=[], done=[])
if RES.get('smoke') not in (None, SMOKE) or RES.get('n') not in (None, N):
    log('results.pkl from other settings or data; starting afresh'); RES = dict(runs=[], subjects=[], sweeps=[], done=[])
RES['smoke'], RES['n'] = SMOKE, N


def save_res(): pickle.dump(RES, open(RES_P + '.tmp', 'wb')); os.replace(RES_P + '.tmp', RES_P)


def window_metrics(P, idx):
    y, p = Y[idx], P[idx]
    return (roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan, average_precision_score(y, p) if y.sum() else np.nan)


def save_model(fm, key, thr):
    if fm['arch'] != 'gnn': return
    torch.save(dict(state=fm['m'].state_dict(), kind=fm['kind'], chs=fm['sub'].chs, st=[x.cpu() for x in fm['st']], thr={k: float(v) for k, v in thr.items()}),
               os.path.join(args.out, 'models', safe(key) + '.pt'))


def run_config(exp, model, nch, seed):
    key = f'{exp}|{model}|{nch}|{seed}'
    if key in RES['done']: return
    t1 = time.time(); P = np.full(N, np.nan, np.float32); thr_of, inner_rows = {}, []
    for k, (trS, teS, inner) in enumerate(FOLDS[exp]):
        ck = os.path.join(args.out, 'ck', safe(key) + f'_f{k}.npz'); te = win_of(teS, True)
        if os.path.exists(ck):
            d = np.load(ck); p = d['p'].astype(np.float32); thr = json.loads(str(d['thr']))
        if not os.path.exists(ck) or len(p) != len(te):
            Pi = np.full(N, np.nan, np.float32); tr = win_of(trS)
            for g in inner:
                fm = fit(model, nch, win_of([s for s in trS if s not in g]), seed); va = win_of(g, True); Pi[va] = predict(fm, va); del fm
            thr = calibrate(Pi, files_of([s for g in inner for s in g]))
            iv = win_of([s for g in inner for s in g], True); thr['inner_auroc'] = window_metrics(Pi, iv)[0]
            fm = fit(model, nch, tr, seed); p = predict(fm, te)
            if exp.startswith('x') and seed == 0: save_model(fm, key, thr)
            del fm; np.savez(ck, p=p.astype(np.float16), thr=json.dumps(thr))
        P[te] = p; inner_rows.append(thr)
        for s in teS: thr_of[s] = thr
    teS_all = [s for _, teS, _ in FOLDS[exp] for s in teS]; te_all = win_of(teS_all, True)
    np.savez_compressed(os.path.join(args.out, 'preds', safe(key) + '.npz'), idx=te_all.astype(np.int32), p=P[te_all].astype(np.float16))
    auroc, auprc = window_metrics(P, te_all)
    sw = sweep(P, files_of(teS_all), THR_GRID); f1s = np.array([r['f1'] if np.isfinite(r['f1']) else -1 for r in sw])
    oracle = sw[int(np.where(f1s == f1s.max())[0][-1])]['t']
    for r in sw: RES['sweeps'].append(dict(exp=exp, model=model, nch=nch, seed=seed, **r))
    base = dict(exp=exp, model=model, nch=nch, seed=seed, auroc=auroc, auprc=auprc, inner_f1=np.nanmean([t['inner_f1'] for t in inner_rows]),
                inner_auroc=np.nanmean([t['inner_auroc'] for t in inner_rows]))
    for op in ['f1'] + [f'fa{b}' for b in BUDGETS] + ['oracle']:
        tot = np.zeros(4); lat_all = []; srows = []
        for s in teS_all:
            t = oracle if op == 'oracle' else thr_of[s][op]; tp, fp, nr, dur, lat = counts(P, FILES_OF[s], t); tot += (tp, fp, nr, dur); lat_all += lat
            ws = win_of([s], True); a, b = window_metrics(P, ws)
            srows.append(dict(exp=exp, model=model, nch=nch, seed=seed, op=op, subject=SUBJ_NAMES[s], threshold=t, tp=tp, fp=fp, nref=nr,
                              hours=dur / 3600, latency=float(np.median(lat)) if lat else np.nan, auroc=a, auprc=b, **scores(tp, fp, nr, dur)))
        if op in ('f1', 'fa4'): RES['subjects'] += srows
        pooled = scores(*tot)
        sm = {f'subj_{k2}': float(np.nanmean([r[k2] for r in srows])) for k2 in ('sens', 'prec', 'f1', 'fp24')}
        RES['runs'].append(dict(base, op=op, threshold=float(np.mean([r['threshold'] for r in srows])), tp=int(tot[0]), fp=int(tot[1]), nref=int(tot[2]),
                                hours=tot[3] / 3600, latency=float(np.median(lat_all)) if lat_all else np.nan,
                                **{f'pooled_{k2}': v for k2, v in pooled.items()}, **sm))
    RES['done'].append(key); save_res()
    r = next(x for x in RES['runs'] if x['exp'] == exp and x['model'] == model and x['nch'] == nch and x['seed'] == seed and x['op'] == 'f1')
    log(f"  {key}: event F1 {r['pooled_f1']:.3f}, sensitivity {r['pooled_sens']:.3f}, FP/24h {r['pooled_fp24']:.1f}, AUROC {auroc:.3f} "
        f"(oracle F1 {next(x['pooled_f1'] for x in RES['runs'] if x['exp'] == exp and x['model'] == model and x['nch'] == nch and x['seed'] == seed and x['op'] == 'oracle'):.3f}; {(time.time() - t1) / 60:.1f} min)")


def write_csv(rows, name):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, name), 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def write_tables():
    write_csv(RES['runs'], 'results_runs.csv'); write_csv(RES['subjects'], 'results_subjects.csv'); write_csv(RES['sweeps'], 'sweeps.csv')


ALL_LEFT = sum(1 for _, todo in STAGES for c in todo if f'{c[0]}|{c[1]}|{c[2]}|{c[3]}' not in RES['done']); T_EXP, N_EXP = time.time(), 0
for name, todo in STAGES:
    left = [c for c in todo if f'{c[0]}|{c[1]}|{c[2]}|{c[3]}' not in RES['done']]
    log(f'stage {name}: {len(todo) - len(left)} of {len(todo)} configurations done')
    for c in left:
        run_config(*c); N_EXP += 1
        log(f'  {N_EXP}/{ALL_LEFT} configurations; about {(time.time() - T_EXP) / N_EXP * (ALL_LEFT - N_EXP) / 60:.0f} min of training left')
    write_tables()
    if args.stop_after == name: sys.exit(0)


# ================================================================== hospital identifiability of the features
def shift_analysis():
    p = os.path.join(args.out, 'shift.csv')
    if os.path.exists(p): return
    rng = np.random.default_rng(0); rows = []
    sel = []
    for s in CHB_S + SIE_S:
        ii = win_of([s], True); ii = ii[Y[ii] == 0]; sel.append(rng.choice(ii, min(len(ii), 1500), replace=False))
    sel = np.concatenate(sel); X = FR[sel]; hosp = DSID[sel]; subj = SUBJ[sel]
    views = {'absolute': X[..., :N_F], 'referenced': np.concatenate([X[..., N_F:], X[..., N_LOG:N_F]], -1), 'both': X}
    for v, Xv in views.items():
        a = []
        for rep in range(3):
            half = {s: rng.random() < 0.5 for s in CHB_S + SIE_S}; m = np.array([half[s] for s in subj])
            if m.all() or (~m).all() or len(set(hosp[m])) < 2 or len(set(hosp[~m])) < 2: continue
            Z = Xv.reshape(len(sel), -1); mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-6
            clf = LogisticRegression(C=0.1, max_iter=500).fit((Z[m] - mu) / sd, hosp[m]); a.append(roc_auc_score(hosp[~m], clf.predict_proba((Z[~m] - mu) / sd)[:, 1]))
        rows.append(dict(kind='hospital_auroc', view=v, auroc=float(np.mean(a)) if a else np.nan, feature='', d_hospital=np.nan))
    names = [f'{lo}-{hi} Hz' for lo, hi in BANDS] + ['line length', 'variance']
    for j, n in enumerate(names):
        for v, col in (('absolute', j), ('referenced', N_F + j)):
            x0, x1 = X[hosp == 0][..., col].mean(1), X[hosp == 1][..., col].mean(1)
            d = abs(x0.mean() - x1.mean()) / (math.sqrt(0.5 * (x0.var() + x1.var())) + 1e-9)
            rows.append(dict(kind='feature_d', view=v, feature=n, d_hospital=float(d), auroc=np.nan))
    write_csv(rows, 'shift.csv'); log('hospital identifiability: ' + ', '.join(f"{r['view']} {r['auroc']:.3f}" for r in rows if r['kind'] == 'hospital_auroc'))


shift_analysis()
if args.stop_after == 'shift': sys.exit(0)


# ================================================================== streaming chain
class StreamingDetector:
    """Causal detector fed with raw EDF data records: band-pass with carried state, decimation, 4-s window buffer,
    features, running-median patient reference, BRIDGE forward pass, 2-of-3 alarm. Times every stage."""
    def __init__(s, E, bundle, quant=False):
        from scipy.signal import sosfilt
        s.sosfilt = sosfilt; s.E = E; s.chs = bundle['chs']; s.n = len(s.chs); s.sos = make_sos(E['fs_in']); s.zi = np.zeros((s.sos.shape[0], s.n, 2))
        s.buf = np.zeros((s.n, 0), np.float32); s.base = np.zeros((W_BASE, s.n, N_LOG), np.float32); s.nb = 0; s.hist = []
        s.kind = bundle['kind']; s.st = [x[s.chs].numpy() for x in bundle['st']]; s.thr = bundle['thr']['f1']
        m = BRIDGE(N_IN[s.kind], Sub(s.chs)); m.load_state_dict(bundle['state']); m = m.cpu().eval()
        s.m = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8) if quant else m
        s.tm = dict(filter=0.0, features=0.0, reference=0.0, model=0.0, alarm=0.0); s.probs = []; s.nw = 0

    def push(s, blk):
        t = time.perf_counter(); x = edf_block(s.E, blk, s.chs); y, s.zi = s.sosfilt(s.sos, x, axis=-1, zi=s.zi)
        s.buf = np.concatenate([s.buf, y[:, ::s.E['dec']].astype(np.float32)], 1); s.tm['filter'] += time.perf_counter() - t
        while s.buf.shape[1] >= WIN:
            s.window(s.buf[:, :WIN]); s.buf = s.buf[:, WIN:]

    @torch.no_grad()
    def window(s, seg):
        t = time.perf_counter(); f, c = window_features(seg[None]); t2 = time.perf_counter(); s.tm['features'] += t2 - t
        s.base[s.nb % W_BASE] = f[0, :, :N_LOG]; s.nb += 1; med = np.median(s.base[:min(s.nb, W_BASE)], 0)
        x = np.concatenate([f[0], f[0, :, :N_LOG] - med], -1); x = (x - s.st[0]) / s.st[1]
        if s.kind == 'rel': x = np.concatenate([x[:, N_F:], x[:, N_LOG:N_F]], -1)
        elif s.kind == 'abs': x = x[:, :N_F]
        t3 = time.perf_counter(); s.tm['reference'] += t3 - t2
        p = float(torch.softmax(s.m(torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(c.astype(np.float32))), -1)[0, 1])
        t4 = time.perf_counter(); s.tm['model'] += t4 - t3
        s.hist = (s.hist + [p >= s.thr])[-N_ALARM:]; s.alarm = sum(s.hist) >= K_ALARM; s.probs.append(p); s.nw += 1
        s.tm['alarm'] += time.perf_counter() - t4


def stream_stage():
    if os.path.exists(os.path.join(args.out, 'stream_timing.csv')): return
    torch.set_num_threads(1); rows, mem, eq = [], [], []
    path = os.path.join(args.data, 'siena', EQUIV_FILE)
    fid = next((f for f, fi in enumerate(FILES) if fi['ds'] == 'siena' and fi['rel'] == EQUIV_FILE), None)
    if not os.path.exists(path) or fid is None: log('streaming stage skipped: test recording missing'); return
    E = open_edf(path, 'siena'); mm = np.memmap(path, dtype='<i2', mode='r', offset=E['H']['hdr_bytes'], shape=(E['nrec'], E['rec_len']))
    for model in ('BRIDGE-R', 'BRIDGE'):
        for nch in (18, 4, 2):
            mp_ = os.path.join(args.out, 'models', safe(f'x_c2s|{model}|{nch}|0') + '.pt')
            if not os.path.exists(mp_): continue
            bundle = torch.load(mp_, map_location='cpu', weights_only=False); sub = Sub(bundle['chs']); net = BRIDGE(N_IN[bundle['kind']], sub).to(dev)
            net.load_state_dict(bundle['state']); net.eval(); st = [x.to(dev) for x in bundle['st']]
            with torch.no_grad():   # offline pipeline in fp32 on the same recording
                x, c = feats(torch.as_tensor(FILE_WIN[fid], device=dev), bundle['kind'], st, sub); p_off = torch.softmax(net(x, c), -1)[:, 1].cpu().numpy()
            for quant in (False, True):
                sd = StreamingDetector(E, bundle, quant)
                for r in range(E['nrec']): sd.push(np.asarray(mm[r:r + 1]))
                ps = np.array(sd.probs); n = min(len(ps), len(p_off)); prec = 'int8' if quant else 'fp32'
                d = np.abs(ps[:n] - p_off[:n]); agree = float(((ps[:n] >= sd.thr) == (p_off[:n] >= sd.thr)).mean())
                eq.append(dict(model=model, nch=nch, precision=prec, windows=n, max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), decision_agreement=agree))
                rows.append(dict(model=model, nch=nch, precision=prec, windows=sd.nw, **{f'ms_{k}': 1000 * v / max(sd.nw, 1) for k, v in sd.tm.items()},
                                 ms_total=1000 * sum(sd.tm.values()) / max(sd.nw, 1), realtime_factor=WIN_S * 1000 / (1000 * sum(sd.tm.values()) / max(sd.nw, 1))))
                params = sum(p.numel() for p in sd.m.parameters()) if not quant else sum(p.numel() for p in BRIDGE(N_IN[sd.kind], Sub(sd.chs)).parameters())
                pbytes = params * (1 if quant else 4)
                state = sd.zi.nbytes + sd.n * WIN * 4 + sd.base.nbytes + sd.n * (N_F + N_LOG) * 4 + len(Sub(sd.chs).pairs) * 4
                macs_feat = sd.n * (5 * WIN * math.log2(WIN) + 20 * WIN) + len(Sub(sd.chs).pairs) * WIN
                d_ = 32; f_ = N_IN[sd.kind]; macs_model = sd.n * f_ * d_ + 2 * (2 * sd.n * d_ * d_ + sd.n * sd.n * d_) + 2 * d_ * d_ + 2 * d_
                mem.append(dict(model=model, nch=nch, precision=prec, params=params, param_kb=pbytes / 1024, state_kb=state / 1024,
                                macs_features=int(macs_feat), macs_model=int(macs_model)))
                log(f"  streaming {model} {nch} ch {prec}: {rows[-1]['ms_total']:.2f} ms per 4-s window, max |diff| to offline {d.max():.2e}, "
                    f"decision agreement {agree:.4f}")
    # int8 accuracy on the whole cross-hospital test set
    q = []
    for model in ('BRIDGE-R', 'BRIDGE'):
        for nch in (18, 4, 2):
            mp_ = os.path.join(args.out, 'models', safe(f'x_c2s|{model}|{nch}|0') + '.pt')
            if not os.path.exists(mp_): continue
            b = torch.load(mp_, map_location='cpu', weights_only=False); sub = Sub(b['chs']); m = BRIDGE(N_IN[b['kind']], sub); m.load_state_dict(b['state']); m = m.cpu().eval()
            mq = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8); te = win_of(SIE_S, True); st = [x.to(dev) for x in b['st']]
            res = {}
            for tag, net in (('fp32', m), ('int8', mq)):
                out = []
                with torch.no_grad():
                    for i in range(0, len(te), 8192):
                        bt = torch.as_tensor(te[i:i + 8192], device=dev); x, c = feats(bt, b['kind'], st, sub)
                        out.append(torch.softmax(net(x.cpu(), c.cpu()), -1)[:, 1])
                P = np.full(N, np.nan, np.float32); P[te] = torch.cat(out).numpy(); tp, fp, nr, dur, _ = counts(P, files_of(SIE_S), b['thr']['f1'])
                res[tag] = dict(auroc=window_metrics(P, te)[0], **scores(tp, fp, nr, dur))
            q.append(dict(model=model, nch=nch, **{f'{t}_{k}': v for t in res for k, v in res[t].items()}))
            log(f"  int8 {model} {nch} ch: AUROC {res['fp32']['auroc']:.4f} -> {res['int8']['auroc']:.4f}, event F1 {res['fp32']['f1']:.3f} -> {res['int8']['f1']:.3f}")
    write_csv(eq, 'stream_equivalence.csv'); write_csv(mem, 'stream_memory.csv'); write_csv(q, 'stream_int8.csv'); write_csv(rows, 'stream_timing.csv')
    torch.set_num_threads(os.cpu_count() or 1)


stream_stage()
if args.stop_after == 'stream': sys.exit(0)


# ================================================================== SzCORE cross-check, timeline, report
def cross_check():
    p = os.path.join(args.out, 'szcore_check.json')
    if os.path.exists(p): return
    try:
        try: import timescoring  # noqa: F401
        except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timescoring'], timeout=240, check=False)
        from timescoring.annotations import Annotation
        from timescoring import scoring
    except Exception as e:
        json.dump(dict(available=False, error=str(e)), open(p, 'w')); log('SzCORE cross-check unavailable:', e); return
    key = 'x_c2s|BRIDGE-R|18|0'; f_ = os.path.join(args.out, 'preds', safe(key) + '.npz')
    if not os.path.exists(f_): return
    d = np.load(f_); P = np.full(N, np.nan, np.float32); P[d['idx']] = d['p'].astype(np.float32)
    t = next(r['threshold'] for r in RES['runs'] if r['exp'] == 'x_c2s' and r['model'] == 'BRIDGE-R' and r['nch'] == 18 and r['seed'] == 0 and r['op'] == 'f1')
    par = scoring.EventScoring.Parameters(toleranceStart=TOL_PRE, toleranceEnd=TOL_POST, minOverlap=0, maxEventDuration=MAX_EVENT, minDurationBetweenEvents=MERGE_GAP)
    agree, n = 0, 0; diffs = []
    for f in files_of(SIE_S):
        L = int(DUR[f]); ref = np.zeros(L, bool); hyp = np.zeros(L, bool)
        for z in SZ:
            if z['fid'] == f: ref[int(round(z['onset'])):int(round(min(z['offset'], L)))] = True
        for s, e in alarm_events(P[FILE_WIN[f]], t, T0[FILE_WIN[f]]): hyp[int(round(s)):int(round(min(e, L)))] = True
        try: sc = scoring.EventScoring(Annotation(ref, 1), Annotation(hyp, 1), par)
        except Exception as e: diffs.append(dict(file=FILES[f]['rel'], error=str(e))); continue
        a, b, c, _ = file_counts(f, P[FILE_WIN[f]], t); n += 1
        if (a, b, c) == (int(sc.tp), int(sc.fp), int(sc.refTrue)): agree += 1
        else: diffs.append(dict(file=FILES[f]['rel'], ours=[a, b, c], timescoring=[int(sc.tp), int(sc.fp), int(sc.refTrue)]))
    json.dump(dict(available=True, recordings=n, identical=agree, differences=diffs[:50]), open(p, 'w'), indent=1)
    log(f'SzCORE cross-check against timescoring: {agree} of {n} recordings identical (TP, FP, reference events)')


def timeline():
    p = os.path.join(args.out, 'timeline.csv')
    if os.path.exists(p): return
    ki, kx = 'in_siena|BRIDGE-R|18|0', 'x_c2s|BRIDGE-R|18|0'
    if not all(os.path.exists(os.path.join(args.out, 'preds', safe(k) + '.npz')) for k in (ki, kx)): return
    Pq = {}
    for k in (ki, kx):
        d = np.load(os.path.join(args.out, 'preds', safe(k) + '.npz')); P = np.full(N, np.nan, np.float32); P[d['idx']] = d['p'].astype(np.float32); Pq[k] = P
    z = next((z for z in SZ if FILES[z['fid']]['ds'] == 'siena' and FILES[z['fid']]['subject'] == 'PN13'), None) or next(z for z in SZ if FILES[z['fid']]['ds'] == 'siena')
    w = FILE_WIN[z['fid']]; t0 = T0[w]; m = (t0 >= z['onset'] - 600) & (t0 <= z['offset'] + 300)
    write_csv([dict(file=FILES[z['fid']]['rel'], onset=z['onset'], offset=z['offset'], t=float(t), p_in=float(a), p_cross=float(b))
               for t, a, b in zip(t0[m], Pq[ki][w][m], Pq[kx][w][m])], 'timeline.csv')


cross_check(); timeline(); write_tables()
make_figures(args.out)
summary = {}
for r in RES['runs']:
    if r['op'] == 'f1' and r['seed'] == 0: summary[f"{r['exp']}|{r['model']}|{r['nch']}"] = dict(f1=r['pooled_f1'], sens=r['pooled_sens'], fp24=r['pooled_fp24'], auroc=r['auroc'])
json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
SUF = '_smoke' if SMOKE else ''
for zname, include in ((f'xhosp_results_v{V}{SUF}.zip', lambda rel: not rel.startswith(('preds', 'ck', 'models'))),
                       (f'xhosp_preds_v{V}{SUF}.zip', lambda rel: rel.startswith(('preds', 'models')))):
    zp = os.path.join(args.out, zname)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(args.out):
            for fn in fs:
                full = os.path.join(root, fn); rel = os.path.relpath(full, args.out)
                if fn.endswith('.zip') or not include(rel): continue
                z.write(full, rel)
    if os.path.isdir(args.drive): shutil.copy(zp, args.drive); log(f'copied {zname} to {args.drive}')
    log(f'{zname}: {os.path.getsize(zp) / 1e6:.1f} MB')
log('done')

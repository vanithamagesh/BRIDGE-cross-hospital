# BRIDGE across hospitals: seizure detection that travels between hospitals and wearable montages

Code for the journal paper *"Seizure Detection Across Hospitals and Wearable Montages: A Patient-Referenced Streaming Detector Evaluated on Pediatric and Adult Scalp EEG"* (authors and venue to be added).

Repository: https://github.com/vanithamagesh/BRIDGE-cross-hospital
Companion conference code: https://github.com/vanithamagesh/BRIDGE-seizure-detection

The study tests seizure detectors in a hospital they were not trained in. Detectors trained on children (CHB-MIT, Boston) are tested on adults (Siena), and the reverse, with 18, 4 or 2 EEG channels. Every threshold comes from the training hospital only, and events are scored with SzCORE. The whole detector, from raw EDF records to the alarm, also runs as a causal streaming chain, timed stage by stage. The key ingredient is a **patient reference**: each log feature minus its running median over the previous 10 minutes of the same recording.

## Main results

| | Event F1 | Sensitivity | False alarms / 24 h | Window AUROC |
|---|---|---|---|---|
| BRIDGE-R, trained on CHB-MIT → tested on Siena | **0.58** | 0.55 | 2.7 | 0.931 |
| BRIDGE-R, trained and tested on Siena (leave-one-subject-out) | 0.58 | 0.53 | 2.4 | 0.942 |
| BRIDGE-A (absolute features), trained on CHB-MIT → tested on Siena | 0.00 | 0.00 | 0.0 | 0.561 |
| BRIDGE-R, trained on Siena → tested on CHB-MIT | 0.35 | 0.33 | 2.5 | 0.930 |
| BRIDGE-R, trained and tested on CHB-MIT (4 subject-wise folds) | 0.47 | 0.52 | 3.1 | 0.946 |

- **Hospital identifiability:** a classifier recognizes the hospital from non-seizure EEG with AUROC 0.98 using the absolute features, but only 0.85 using the referenced ones.
- **Two temporal channels** (T7-P7, T8-P8) keep the cross-hospital F1 at 0.55 (CHB-MIT → Siena) and 0.44 (Siena → CHB-MIT).
- **Streaming:** 3.3–5.6 ms per 4-s window on one CPU thread, 19–170 kB of state and 6820 parameters. The window decisions are identical to the offline pipeline.
- **Scoring check:** our SzCORE event scoring matches the `timescoring` package on all 41 Siena recordings.

All numbers are in `results/`.

## Data

Both databases are open access and are downloaded automatically from the PhysioNet AWS open-data mirror.

| Database | Subjects | Recordings used | Hours | Seizures | License |
|---|---|---|---|---|---|
| CHB-MIT Scalp EEG Database v1.0.0 | 23 children and adolescents (1.5–22 years) | 682 of 686 | 976.6 | 185 | ODC-BY 1.0 |
| Siena Scalp EEG Database v1.0.0 | 14 adults (20–71 years) | 41 | 141.0 | 47 | CC BY 4.0 |

Siena is converted to the same 18 bipolar channels as CHB-MIT. Its seizure lists contain typos, so the script reads the recording start from the EDF headers and logs every correction in `dataset.json`.

## How to run

**Get the code:**

```
git clone https://github.com/vanithamagesh/BRIDGE-cross-hospital.git
cd BRIDGE-cross-hospital
pip install -r requirements.txt
```

**Google Colab (recommended):** Runtime → Change runtime type → A100 GPU, High-RAM. Then run:

```
!git clone https://github.com/vanithamagesh/BRIDGE-cross-hospital.git
%cd BRIDGE-cross-hospital
!pip install -q timescoring              # the other packages are preinstalled on Colab
!python bridge_cross_hospital.py --smoke # about 10 min: checks the whole pipeline on 3 subjects per hospital
!python bridge_cross_hospital.py         # full run: about 68 GB download, deleted after use; about 1 h on an A100
```

- **On a T4 GPU:** add `--no_raw` to skip the raw-signal CNN and its 19 GB of signals.
- **If a run stops:** start it again with the same command. Processed files, finished folds and stages are loaded from checkpoints.
- **Outputs:** written to `/content/xhosp_out/`, with the zip `xhosp_results_v1_1.zip`.

**Figures from stored results (no data or GPU needed):**

```
python bridge_cross_hospital.py --figs results      # Figs. 2-10 of the paper into results/figures/
python figures/make_fig1_chain.py results/figures   # Fig. 1 (chain, protocol, channel sets)
```

## What the script does

1. **Data:** downloads every recording and computes features in parallel. The chain is causal: a band-pass filter whose state is carried across blocks, decimation to 256 Hz, then non-overlapping 4-s windows. Each window gives 7 log band powers, log line length, log variance, kurtosis, zero-crossing rate and inter-channel correlations.
2. **Patient reference:** subtracts the running median of the previous 150 windows from the log features. It needs no labels and no data from the new hospital.
3. **Detectors:**
   - BRIDGE in three versions: BRIDGE with both views, BRIDGE-R with the referenced view only, and BRIDGE-A with the absolute view only.
   - Baselines: channel Transformer, MLP, logistic regression, gradient-boosted trees (also referenced-only, GBDT-R) and a raw-signal 1-D CNN.
4. **Protocols:**
   - In-hospital: Siena leave-one-subject-out; CHB-MIT 4 subject-wise folds.
   - Cross-hospital: train on all of one database, test on all of the other.
   - Channel sets: 18, 4 and 2. The BRIDGE variants across hospitals are repeated with 3 seeds.
5. **Thresholds:** chosen on inner subject-wise folds of the training hospital only: maximum event F1, and false-alarm budgets of 1, 4 and 12 per 24 h. The oracle threshold on the test hospital is reported for analysis only.
6. **Scoring:**
   - Alarm when 2 of the last 3 windows are positive.
   - SzCORE event scoring: alarms from 30 s before onset to 60 s after offset count as detections, events closer than 90 s are merged, and events longer than 5 min are split.
   - Also reported: false alarms per 24 h, detection latency, and window AUROC/AUPRC.
7. **Analyses:** hospital identifiability of the features, a streaming implementation timed per stage (fp32 and int8) and checked window by window against the offline pipeline, then figures and tables.

## Repository

| Path | Content |
|---|---|
| `bridge_cross_hospital.py` | The whole study in one script |
| `figures/make_fig1_chain.py` | Fig. 1 of the paper |
| `results/results_runs.csv` | One row per experiment, detector, channel set, seed and operating point |
| `results/results_subjects.csv` | Per-subject event scores |
| `results/sweeps.csv` | Sensitivity, precision, F1 and false alarms over a threshold sweep |
| `results/shift.csv` | Hospital identifiability and per-feature differences |
| `results/stream_*.csv` | Streaming timing, memory, int8 effect and agreement with the offline pipeline |
| `results/dataset.json` | Subjects, hours, seizures, ages, skipped files and annotation corrections |

## Citation

Please cite the paper (reference to be added after acceptance) and the two databases:

- A. H. Shoeb, "Application of machine learning to epileptic seizure onset detection and treatment," Ph.D. dissertation, MIT, 2009.
- P. Detti, G. Vatti, and G. Zabalo Manrique de Lara, "EEG synchronization analysis for seizure prediction: A study on data of noninvasive recordings," *Processes*, vol. 8, no. 7, Art. no. 846, 2020.
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.

## License

MIT License (see `LICENSE`). The EEG data keep their own licenses (CHB-MIT: ODC-BY 1.0; Siena: CC BY 4.0).

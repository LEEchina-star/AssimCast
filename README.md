# AssimCast

**Coarse-to-fine, pixel-space diffusion forecasting of tropical-cyclone surface
winds, with training-free multi-satellite assimilation.**

AssimCast takes an operational coarse analysis (0.25°, e.g. ERA5/GFS) and produces
six-hour, storm-centered 0.06° ocean-surface wind fields that recover the eyewall,
radius of maximum winds and high-wind tail that coarse reanalyses smooth away. The
same frozen model is then guided at inference by real SAR / SMAP / AMSR2 winds
**without any retraining**.

The model has three stages (see the manuscript):

| Stage | Role | Learned from |
|---|---|---|
| **A — Motion** | `CenterHead` predicts the 6 h vortex displacement; a continuous tanh whole-state displacement backward-warps the coarse analysis (the vortex follows the storm, the outer field blends to the environmental flow). | 30 years of coarse ERA5 (1993–2018) |
| **B — Super-resolution** | A conditional pixel-space diffusion learns the motion-compensated coarse→fine increment `x0 = HAFS_f006 − era_adv` (eyewall, RMW, high-wind tail, asymmetry, 0.06° sharpness). | HAFS high-resolution forecasts (**2023 season only**) |
| **C — Training-free assimilation** | The frozen Stage-B model is steered by real satellite overpasses through a Gaussian-footprint observation operator (posterior guidance in pixel space); no model weights are touched. | Real SAR / SMAP / AMSR2 at inference |

**Conditioning channels (10).** fine history at τ−18, −12, −6 h (6 channels, ERA5
upsampled to the 0.06° window), the Stage-A advected τ field (2), and a 32°
native-resolution coarse environment (2). The grid is 336 × 336 at 0.06°
(≈20.1° window).

> **Data-splitting note.** The Stage-B fine-tuning uses only the 2023 HAFS season.
> The full 2024 season (1,446 six-hour samples) is held out, and 2025/2026 are
> used solely for the training-free deployment/assimilation test. HAFS itself is
> an operational product that continues to run; only the 2023 season was archived
> for training. Satellite winds (SAR, SMAP, AMSR2) are used through 2026.

## Repository layout

```
AssimCast/
├── model/                  # core library (flat, added to sys.path)
│   ├── diffusion_model.py  # SimpleUNet backbone (U-Net + attention, timestep embedding)
│   ├── tc_model.py         # create_tc_model(): backbone + CenterHead
│   ├── diffusion_process.py# GaussianDiffusion (DDPM/DDIM), EMA
│   ├── dpm_solver.py       # alternative sampler
│   ├── v5_warp.py          # Stage A whole-field continuous warp (warp_field, vortex_weight)
│   ├── hafs_dataset.py     # HAFSCoreDataset (0.06° storm window, normalisation)
│   └── da_v6_footprint.py  # Stage C Gaussian-footprint observation operator
├── train.py                # Stage B coarse-to-fine fine-tuning (warm-starts Stage A)
├── predict.py              # six-hour 0.06° forecast from a coarse analysis
├── predict_assimilate.py   # Stage C training-free multi-satellite assimilation
├── preprocessing/          # HAFS/ERA5/GFS fetch + sample/observation builders
├── scoring/                # independent evaluation (satellite, IBTrACS, external baselines)
├── figures/                # plotting scripts that reproduce the manuscript figures
└── configs/config_assimcast.yaml
```

## Installation

```bash
pip install -r requirements.txt
```
CUDA (NVIDIA), Apple MPS and CPU are all supported (device is auto-selected).

## Data

Datasets are large and are not redistributed here. Set the package root and the
external-data roots through environment variables (the preprocessing scripts read
these):

```bash
export ASSIMCAST_ROOT=/path/to/AssimCast   # defaults to this repository
export ERA5_ROOT=/path/to/ERA5_monthly_nc
export IBTRACS_PATH=/path/to/IBTrACS.since1980.v04r01.nc
```

* **HAFS** high-resolution forecasts — fetched with `preprocessing/fetch_hafs.py`,
  `fetch_hafs_lead.py`, `fetch_parent_lead.py` (NOAA operational archive).
* **ERA5** 0.25° 10 m winds / MSL — ECMWF CDS.
* **GFS** 0.25° analyses (deployment test) — `preprocessing/fetch_gfs_10m.py`.
* **SAR** inner-core winds — CyclObs / Ifremer collection; `preprocessing/make_sar_global.py`.
* **SMAP / AMSR2** radiometer winds — Remote Sensing Systems TC-winds;
  `preprocessing/make_rss_global.py` (`--src smap|amsr2`).

Build the staged training/validation samples and the causal observation manifest:

```bash
python preprocessing/build_hafs_core.py
python preprocessing/build_parent_core_v6.py     # 336 x 336, 0.06 deg parent windows
python preprocessing/build_multibasin.py         # multi-basin coarse samples
python preprocessing/build_da_causal_global.py   # leakage-free assimilation manifest
```

## Training

Stage A is trained on coarse ERA5; Stage B is the coarse-to-fine fine-tuning and
warm-starts the converged Stage-A motion/geometry weights (provide the checkpoint
or it will train from scratch):

```bash
python train.py --data $ASSIMCAST_ROOT/data/hafs_parent_v6 \
                --cfg configs/config_assimcast.yaml \
                --ckpt runs/stage_a/checkpoints/best.pt
```

All Stage-B loss weights (track, gradient, spectrum, high-wind, anti-peak-shaving,
asymmetric under-prediction penalty) are command-line flags with the final values
shown in `configs/config_assimcast.yaml`.

## Forecast and training-free assimilation

```bash
# plain six-hour forecast
python predict.py --ckpt runs/hafs_v6/checkpoints/best.pt --sets test

# forecast guided by real SAR + SMAP + AMSR2 (no retraining)
python predict_assimilate.py --ckpt runs/hafs_v6/checkpoints/best.pt \
                             --assim sar,smap,amsr2 --steps 50
```

The causal design uses only whole overpasses acquired up to the analysis time; a
later overpass crossing the eyewall is withheld and used as the independent arbiter.

## Evaluation and figures

```bash
python scoring/score_da_future_obs_global.py   # leakage-free future-overpass RMSE
python scoring/hafs_gate_ibtracs.py            # HAFS vs ERA5 structure gate
python scoring/score_structure_radii.py        # RMW / wind radii
python scoring/score_external_baselines.py     # Pangu-Weather / GraphCast / FourCastNet
```
Bootstrap confidence intervals use `scoring/bootstrap_ci.py`. Figure scripts in
`figures/` read the `results/*.json|csv` outputs and write vector PDF figures.

## What is and is not included

* Included: the complete model, training, inference, training-free assimilation,
  evaluation and figure code.
* Not included: raw datasets, trained checkpoints, prediction `.npz` and cached
  results — these are generated with the commands above or available from the
  corresponding archives.
* Satellite winds used are **SAR, SMAP and AMSR2 only**. No GNSS-R, simulated
  microwave winds, or tide-gauge/storm-surge data are part of this model.

## License

See [LICENSE](LICENSE).

## Citation

Manuscript in submission at *Nature Communications*; please cite the published
version once available (citation to be added).

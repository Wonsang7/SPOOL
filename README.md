# SPOOL: photon-efficient quantitative imaging

Reference simulation code for **Spatially pooling photon information enables photon-efficient quantitative imaging**.

SPOOL (Spatially Pooled Optical Observation Likelihood) jointly estimates source-space component amplitudes from photon-count data using a known point-spread function (PSF) and a contrast dictionary. This repository reproduces the homogeneous and heterogeneous multi-emitter FLIM benchmarks and includes the dual-labeled FLIM and hyperspectral analysis drivers.

## Implemented manuscript settings

- 15 emitters on a 100 x 100 pixel field
- 50 nm sampling and a pixel-integrated 200 nm-FWHM Gaussian PSF
- 256 time bins at 0.0977 ns per bin and a 150 ps-FWHM Gaussian IRF
- ground-truth lifetimes of 2 and 4 ns
- lifetime-agnostic reconstruction dictionary: 11 bases from 1.0 to 5.0 ns in 0.4 ns steps
- foreground signal-to-background ratio of 50 at every photon level
- eight Poisson trials per photon level
- SPOOL: 50 iterations with damping exponent 0.9

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is optional. If a CUDA-enabled PyTorch installation is available, the iterative reconstruction uses the GPU; otherwise it falls back to NumPy on the CPU.

Optional GPU acceleration can be installed with:

```bash
pip install -r requirements-optional.txt
```

## Run

```bash
python run_multiemitter_benchmark.py
```

Outputs are written to `results/multiemitter/`. To use another location:

```bash
SPOOL_RESULTS_DIR=/path/to/results python run_multiemitter_benchmark.py
```

The default run includes pixel-wise Poisson MLE, NC-PCA followed by NNLS, phasor analysis, and SPOOL.

### Single-emitter, oracle, and supplementary controls

```bash
python oracle_pooled_control.py
python run_iteration_selection.py
python sensitivity_sweep_pipeline.py
python run_sequential_controls.py
python runtime_benchmark.py
python run_bead_microtubule.py
```

These scripts implement the manuscript settings: a pixel-integrated 200 nm-FWHM
Gaussian PSF at 50 nm sampling, a fixed single-emitter background of
`1e-4` counts per pixel per time bin, a lifetime-agnostic 11-component
dictionary, 50 SPOOL iterations, and damping exponent 0.9. The sensitivity
driver generates its own condition-specific background levels for the stated
signal-to-background-ratio sweep.

## Files

- `run_multiemitter_benchmark.py`: canonical manuscript benchmark runner
- `flim_sim_ncpca.py`: forward model, scene generation, NC-PCA, phasor, and sequential baseline utilities
- `run_dual_labeled_flim.py`: dual-labeled experimental FLIM analysis
- `run_hyperspectral.py`: experimental hyperspectral analysis
- `run_bead_microtubule.py`: experimental bead and single-labeled microtubule photon-thinning analysis
- `oracle_pooled_control.py`: single-emitter Monte Carlo, CRBs, 3x3 binning, and oracle pooled MLE
- `run_iteration_selection.py`: fixed-iteration selection and PSF-perturbation analysis
- `sensitivity_sweep_pipeline.py`: background, PSF, and background-rate mismatch sweeps
- `run_sequential_controls.py`: Wiener, total-variation, and Richardson-Lucy sequential controls
- `bias_ablation.py`: dictionary-spacing and early-stopping bias ablation
- `runtime_benchmark.py`: f64/f32 GPU runtime, f64 CPU runtime, and peak GPU-memory benchmark

## Reproducibility notes

Generation and reconstruction dictionaries are deliberately separated. Photon counts are generated using the two true lifetimes, whereas every dictionary-based reconstruction uses the same lifetime-agnostic 11-component dictionary. Random seeds are fixed in the runner. Numerical results may differ slightly between CPU and GPU backends because of floating-point arithmetic.

## Experimental data

The synthetic benchmark requires no external data. The experimental photon-count data are not redistributed here.

## Citation

Citation information will be added when the manuscript is publicly available.

## License

The SPOOL code in this repository is released under the MIT License.

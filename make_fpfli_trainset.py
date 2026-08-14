"""
make_fpfli_trainset.py
======================
Builds Training_data.mat for FPFLI's local lifetime estimator (LLE), in the
format its own loader expects and -- critically -- in the representation its own
inference adapter feeds it.

WHY THE FIRST ATTEMPT FAILED
    Training on single-pixel raw-count curves against a target in nanoseconds
    produced a network that was accurate in isolation (RMSE 0.80 ns, within 7%
    of the information floor) yet gave 5-7 ns RMSE inside the benchmark. The
    reason is that OfficialFPFLI._predict_local_lifetime does three things to
    the cube before the LLE ever sees it:

        blocks = cube.reshape(...).sum(axis=(1,3))      # 8x8 SPATIAL POOLING
        flat   = flat / flat.max(axis=1, keepdims=True) # MAX-NORMALIZATION
        tau    = lle(...) * (dt_ns * 100.0)             # OUTPUT RESCALING

    None of the three was present in the training data. The third alone
    multiplies every prediction by 3.9 (at dt = 0.039 ns) or 9.8 (at 0.0977 ns),
    which is the whole of the observed discrepancy. This script reproduces all
    three, so that what the network is trained on is what it is asked to
    predict.

WHAT THE LOADER EXPECTS  (utils.load_data)
    y        (N, 256) float   8x8-pooled, max-normalized decay curves
    I        (N, 256) float   IRF, one row per sample, max-normalized
    tau_ave  (N,)     float   target in the LLE's internal units, tau/(dt*100)

DT_ADAPTER MUST MATCH THE BENCHMARK
    The output rescaling uses the dt_ns given to OfficialFPFLI. If the benchmark
    constructs the adapter as OfficialFPFLI(repo, dt_ns=fs.DT_NS, ...) then
    DT_ADAPTER below must equal fs.DT_NS. A mismatch rescales every prediction
    by their ratio and invalidates the comparison outright.

SEEDS
    Scenes are drawn from seeds 3000+, disjoint from the evaluation seeds
    2000-2007.

    python make_fpfli_trainset.py
"""

import os
import numpy as np
from scipy.special import erf, erfc
from scipy.signal import fftconvolve
from scipy.io import savemat

import flim_sim_ncpca as fs

# ============================ CONFIG ============================
HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_MAT   = os.environ.get("FPFLI_TRAINING_DATA",
                           os.path.join(HERE, "Training_data.mat"))
N_SAMPLES = 120_000          # pooled blocks in the final, stratified set
N_BEADS   = 15

SEED_BASE = 3000             # evaluation uses 2000-2007

TAU_RANGE = (1.0, 5.0)       # matches the K = 11 agnostic dictionary
PL_RANGE  = (25.0, 3000.0)   # photons per bead; wider than the sweep's 50-800
                             # so that the bright strata fill without replacement

# --- adapter contract: these must mirror OfficialFPFLI ----------------------
PATCH_SIZE            = 8       # OfficialFPFLI.patch_size
LOCAL_COUNT_THRESHOLD = 50.0    # OfficialFPFLI.local_count_threshold
PIXEL_THRESHOLD       = 3.0     # OfficialFPFLI.pixel_threshold
DT_ADAPTER            = None    # None -> take fs.DT_NS; must equal the dt_ns
                                # passed to OfficialFPFLI in the benchmark

# photon-count strata, in counts per POOLED BLOCK (64 pixels), not per pixel
STRATA = [(50, 100), (100, 200), (200, 400), (400, 900), (900, 2000), (2000, 8000)]
OVERSAMPLE = 2               # collect this many times N_SAMPLES before thinning

# forward model, matched to single_bead_crb_mc.py and the benchmark
PIXEL_NM      = 50.0
PSF_FWHM_NM   = 200.0
BG_PER_BIN    = 1e-4
PSF_INTEGRATE = True
# ===============================================================


def gaussian_psf(sigma_px, integrate=True):
    r = max(int(np.ceil(4 * sigma_px)), 1)
    a = np.arange(-r, r + 1)
    if integrate:
        w = 0.5 * (erf((a + 0.5) / (np.sqrt(2) * sigma_px))
                   - erf((a - 0.5) / (np.sqrt(2) * sigma_px)))
        p = np.outer(w, w)
    else:
        xx, yy = np.meshgrid(a, a)
        p = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_px ** 2))
    return p / p.sum()


def build_decay_basis(taus, N_T, dt, t0, sig):
    """IRF-convolved, window-normalized exponential decays, Eq. (S2)."""
    t = np.arange(N_T) * dt
    tau = np.atleast_1d(np.asarray(taus, dtype=np.float64))[:, None]
    tt = t[None, :]
    lam = 1.0 / tau
    arg = (t0 + lam * sig ** 2 - tt) / (np.sqrt(2) * sig)
    ex = np.clip(0.5 * lam ** 2 * sig ** 2 - lam * (tt - t0), -700, 700)
    d = np.clip(0.5 * lam * np.exp(ex) * erfc(arg), 0.0, None)
    s = d.sum(axis=1, keepdims=True)
    return d / np.where(s > 0, s, 1.0)


def pool_blocks(arr, P):
    """Sum P x P spatial blocks. arr is (H, W) or (H, W, T)."""
    H, W = arr.shape[:2]
    hp, wp = (H // P) * P, (W // P) * P
    a = arr[:hp, :wp]
    if a.ndim == 2:
        return a.reshape(hp // P, P, wp // P, P).sum(axis=(1, 3))
    return a.reshape(hp // P, P, wp // P, P, a.shape[2]).sum(axis=(1, 3))


def main():
    N_T = int(fs.N_TIME_BINS)
    dt = float(fs.DT_NS)
    t0 = float(fs.IRF_PEAK_BIN) * dt
    sig = float(fs.IRF_FWHM_NS) / 2.3548200450309493
    sigma_px = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM
    psf = gaussian_psf(sigma_px, PSF_INTEGRATE)
    dt_adapter = dt if DT_ADAPTER is None else float(DT_ADAPTER)
    scale = dt_adapter * 100.0          # the adapter multiplies the LLE output by this

    if N_T != 256:
        raise SystemExit(f"FPFLI requires 256 time bins; this dataset has {N_T}.")

    fs.PSF, fs.BACKGROUND_RATE = psf, BG_PER_BIN

    # the IRF row, max-normalized, exactly as gaussian_irf() and predict() produce it
    t = np.arange(N_T) * dt
    irf_row = np.exp(-0.5 * ((t - t0) / sig) ** 2)
    irf_row = (irf_row / irf_row.max()).astype(np.float32)

    print(f"forward model : sigma_PSF {sigma_px:.3f} px, {psf.shape[0]}x{psf.shape[0]} kernel, "
          f"h(0) {psf.max():.5f}")
    print(f"                bin width {dt:.4f} ns, window {N_T*dt:.2f} ns, "
          f"IRF FWHM {fs.IRF_FWHM_NS*1000:.0f} ps, t0 {t0:.3f} ns")
    print(f"                background {BG_PER_BIN:.3g}/bin = {BG_PER_BIN*N_T:.4f} photons/px")
    print(f"adapter       : {PATCH_SIZE}x{PATCH_SIZE} pooling, max-normalized input,")
    print(f"                target = tau / (dt_ns * 100) with dt_ns = {dt_adapter:.4f}"
          f"  -> divide by {scale:.3f}")
    print(f"                blocks kept above {LOCAL_COUNT_THRESHOLD:g} counts, "
          f"pixels above {PIXEL_THRESHOLD:g}")
    if DT_ADAPTER is not None and abs(dt_adapter - dt) > 1e-9:
        print(f"  !! DT_ADAPTER ({dt_adapter}) differs from fs.DT_NS ({dt}); this is only")
        print(f"     correct if the benchmark passes dt_ns={dt_adapter} to OfficialFPFLI.")
    print(f"strata (counts per pooled block): {STRATA}")
    print(f"seeds         : {SEED_BASE}+ (evaluation uses 2000-2007)\n")

    keep_true = (fs.D_BASES, fs.TAUS_NS, fs.K_SPECIES)
    Y_all, T_all, N_all = [], [], []
    n_have, scene = 0, 0
    target_pool = N_SAMPLES * OVERSAMPLE
    P = PATCH_SIZE

    try:
        while n_have < target_pool:
            rng = np.random.default_rng(SEED_BASE + scene)
            taus = np.sort(rng.uniform(*TAU_RANGE, 2))
            pl = float(np.exp(rng.uniform(*np.log(PL_RANGE))))

            fs.TAUS_NS = taus
            fs.K_SPECIES = 2
            fs.D_BASES = build_decay_basis(taus, N_T, dt, t0, sig)

            A_true, coords = fs.generate_scene(n_beads=N_BEADS, photons_per_bead=pl,
                                               rng=rng, n_species0=None)
            Y, Lam = fs.simulate_measurement(A_true, rng)

            # detector-space amplitude maps and the per-pixel mean lifetime they imply
            Ab = np.stack([fftconvolve(A_true[k], psf, mode="same") for k in range(2)])
            tot = Ab.sum(axis=0)
            num = (taus[:, None, None] * Ab).sum(axis=0)

            # the adapter masks dim pixels before pooling; reproduce that
            keep_px = Y.sum(axis=2) > PIXEL_THRESHOLD
            Ym = Y * keep_px[:, :, None]

            blocks  = pool_blocks(Ym, P)                     # (nb, nb, T)
            btot    = pool_blocks(tot * keep_px, P)          # amplitude in the block
            bnum    = pool_blocks(num * keep_px, P)
            btau    = np.divide(bnum, btot, out=np.zeros_like(btot), where=btot > 1e-12)
            bcounts = blocks.sum(axis=2)

            m = (bcounts > LOCAL_COUNT_THRESHOLD) & (btot > 1e-12)
            if m.sum() == 0:
                scene += 1
                continue

            Y_all.append(blocks[m].astype(np.float32))
            T_all.append(btau[m].astype(np.float32))
            N_all.append(bcounts[m].astype(np.float32))
            n_have += int(m.sum())
            scene += 1
            if scene % 100 == 0:
                print(f"  {scene:>5} scenes -> {n_have:>8} blocks "
                      f"({100*n_have/target_pool:.0f}% of pool target)")
    finally:
        fs.D_BASES, fs.TAUS_NS, fs.K_SPECIES = keep_true

    y = np.concatenate(Y_all)
    tau_ave = np.concatenate(T_all)
    npx = np.concatenate(N_all)
    print(f"\npool: {len(y)} pooled blocks from {scene} scenes")
    print(f"  counts per block {npx.min():.0f}-{npx.max():.0f} (median {np.median(npx):.0f})")

    # ---------------------------------------------------- stratify
    per_stratum = N_SAMPLES // len(STRATA)
    rng_s = np.random.default_rng(SEED_BASE - 2)
    ys, ts, ns = [], [], []
    print(f"\nstratified sampling ({per_stratum} per stratum):")
    for lo, hi in STRATA:
        idx = np.flatnonzero((npx >= lo) & (npx < hi))
        if len(idx) == 0:
            print(f"  {lo:>5}-{hi:<5}: EMPTY -- widen PL_RANGE or raise OVERSAMPLE")
            continue
        repl = len(idx) < per_stratum
        take = rng_s.choice(idx, per_stratum, replace=repl)
        ys.append(y[take]); ts.append(tau_ave[take]); ns.append(npx[take])
        print(f"  {lo:>5}-{hi:<5}: {len(idx):>8} available, {per_stratum} taken"
              f"{'   WITH REPLACEMENT' if repl else ''}")
    if not ys:
        raise SystemExit("no stratum was populated")
    y = np.concatenate(ys); tau_ave = np.concatenate(ts); npx = np.concatenate(ns)

    idx = np.random.default_rng(SEED_BASE - 1).permutation(len(y))
    y, tau_ave, npx = y[idx], tau_ave[idx], npx[idx]

    # ---------------------------------------------------- adapter representation
    tau_ns = tau_ave.copy()                       # keep for reporting
    mx = y.max(axis=1, keepdims=True)
    y = np.divide(y, mx, out=np.zeros_like(y), where=mx > 0)     # max-normalize
    tau_ave = tau_ave / scale                                     # LLE-internal units

    I = np.tile(irf_row, (len(y), 1))

    print(f"\nfinal set: {len(y)} blocks")
    print(f"  tau (ns)        {tau_ns.min():.2f}-{tau_ns.max():.2f}  "
          f"(median {np.median(tau_ns):.2f})")
    print(f"  tau (LLE units) {tau_ave.min():.4f}-{tau_ave.max():.4f}  "
          f"(divided by {scale:.3f})")
    print(f"  input          max-normalized, peak {y.max():.3f}, "
          f"mean nonzero bins {np.mean((y > 0).sum(1)):.0f} of {N_T}")
    print(f"  counts/block   {npx.min():.0f}-{npx.max():.0f} (median {np.median(npx):.0f})")
    for lo, hi in STRATA:
        f = ((npx >= lo) & (npx < hi)).mean()
        print(f"    {lo:>5}-{hi:<5}: {100*f:5.1f}%")

    savemat(OUT_MAT, {"y": y, "I": I, "tau_ave": tau_ave.reshape(1, -1),
                      "tau_ns": tau_ns.reshape(1, -1),
                      "counts_per_block": npx.reshape(1, -1),
                      "dt_adapter": dt_adapter, "scale": scale},
            do_compression=True)
    print(f"\n-> {OUT_MAT}  ({os.path.getsize(OUT_MAT)/1e6:.0f} MB)")
    print("   place next to Training_LLE.py and run it unchanged")
    print(f"\n   REMEMBER: the benchmark must construct the adapter with")
    print(f"   OfficialFPFLI(repo, dt_ns={dt_adapter}, ...) or the {scale:.3f}x")
    print(f"   output rescaling will not match what this training set assumes.")


if __name__ == "__main__":
    main()

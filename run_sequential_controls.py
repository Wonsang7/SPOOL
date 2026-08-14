"""
supple1_two_stage_figure.py
===========================
Sequential-control study (Supplementary S4, Fig. S1) at the 50-iteration
protocol, with the data path CLONED from sim_homo_hetero_panels_gpu.py so
that the pixel-wise MLE and SPOOL curves reproduce the main homogeneous
benchmark exactly: same seeds (2000+trial), same scene generator, same
forward model, same estimator implementations, same readout and fallback.

A built-in GATE compares this run's MLE and SPOOL curves against the main
it50 benchmark values before anything is rendered. Two background modes are
tried automatically (the one open question between script generations):
    "fixed" : fs.BACKGROUND_RATE = 1e-4 for every photon level
    "snb"   : per-level rate held at a constant signal-to-background ratio
              (the per-level values recorded by the previous control run)
Whichever mode passes the gate is used for the controls; if neither passes,
the per-level differences are printed and NOTHING is rendered.

Controls (identical Y per trial, unmixed ONCE, deconvolved three ways):
    Wiener : fs.wiener_deconvolve, SNR=50
    TV     : fs.tv_deconvolve, lam=0.02, 100 steps
    RL     : same multiplicative update, same PSF, eta=0.9, 50 iterations

Place NEXT TO flim_sim_ncpca.py and run:
    python supple1_two_stage_figure.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import time

import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flim_sim_ncpca as fs

# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("SPOOL_SEQUENTIAL_RESULTS_DIR",
                         os.path.join(HERE, "results", "sequential_controls"))

TAU_GEN       = [2.0, 4.0]      # generation dictionary of the main it50 benchmark
N_BEADS       = 15
PHOTON_LEVELS = [50, 100, 200, 400, 800]
N_TRIALS      = 8
N_ITER        = 50              # joint AND RL control AND pixel-wise MLE
ETA           = 0.9
FALLBACK_TAU  = 3.0             # detection-failure fallback = mean of TAU_GEN
PIXEL_NM      = 50.0
PSF_FWHM_NM   = 200.0
FG_FRAC       = 0.15
EPS           = 1e-9

# background candidates (auto-selected by the gate)
BG_FIXED   = 1e-4
BG_SNB     = {50: 9.366041902188683e-05, 100: 1.8732083804377366e-04,
              200: 3.746416760875473e-04, 400: 7.492833521750946e-04,
              800: 1.4985667043501893e-03}

# main it50 homogeneous benchmark values -- THE GATE
GATE_MLE  = [0.8930, 0.9089, 0.7022, 0.5206, 0.3446]
GATE_PROP = [0.4108, 0.2886, 0.2023, 0.1487, 0.1122]
GATE_ATOL = 5e-4
# ===============================================================

# ---------------------------------------------------------------- backend
USE_TORCH = False
DEVICE = None
try:
    import torch
    if torch.cuda.is_available():
        USE_TORCH = True; DEVICE = torch.device("cuda")
        print(f"Backend: PyTorch / GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Backend: NumPy / CPU")
except ImportError:
    print("Backend: NumPy / CPU (torch not installed)")


def to_dev(a):
    if USE_TORCH:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=DEVICE)
    return np.asarray(a, dtype=np.float64)


def to_np(a):
    if USE_TORCH and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# ---------------------------------------------------------------- forward model
def gaussian_psf(sigma_px):
    r = max(int(np.ceil(4 * sigma_px)), 1)
    a = np.arange(-r, r + 1)
    w = 0.5 * (erf((a + 0.5) / (np.sqrt(2) * sigma_px))
               - erf((a - 0.5) / (np.sqrt(2) * sigma_px)))
    p = np.outer(w, w)
    return p / p.sum()


def patch_module():
    """TAU_GEN + pixel-integrated PSF into the phantom module, as the main
    it50 runner does."""
    sigma_px = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM
    fs.PSF = gaussian_psf(sigma_px)
    fs.TAUS_NS = list(TAU_GEN)
    fs.D_BASES = fs.make_decay_bases(TAU_GEN, fs.N_TIME_BINS, fs.DT_NS, fs.IRF_FWHM_NS)
    fs.K_SPECIES = len(TAU_GEN)
    print(f"phantom: tau {TAU_GEN} ns, sigma {sigma_px:.3f} px pixel-integrated, "
          f"h(0) {fs.PSF.max():.5f}, fallback {FALLBACK_TAU} ns")


# ---------------------------------------------------------------- FFT ops (cloned)
def make_otf(psf, shape):
    H, W = shape; ph, pw = psf.shape
    fh, fw = H + ph - 1, W + pw - 1
    pad = np.zeros((fh, fw)); pad[:ph, :pw] = psf
    pad_f = np.zeros((fh, fw)); pad_f[:ph, :pw] = psf[::-1, ::-1]
    if USE_TORCH:
        otf = torch.fft.rfft2(to_dev(pad), s=(fh, fw))
        otf_adj = torch.fft.rfft2(to_dev(pad_f), s=(fh, fw))
    else:
        otf = np.fft.rfft2(pad, s=(fh, fw)); otf_adj = np.fft.rfft2(pad_f, s=(fh, fw))
    return dict(otf=otf, otf_adj=otf_adj, fshape=(fh, fw), out=(H, W),
                off=((ph - 1)//2, (pw - 1)//2))


def _fft_op(A, o, adjoint=False):
    fh, fw = o["fshape"]; H, W = o["out"]; y0, x0 = o["off"]
    G = o["otf_adj"] if adjoint else o["otf"]
    if USE_TORCH:
        F = torch.fft.rfft2(A, s=(fh, fw))
        full = torch.fft.irfft2(F * G.unsqueeze(0), s=(fh, fw))
    else:
        F = np.fft.rfft2(A, s=(fh, fw), axes=(1, 2))
        full = np.fft.irfft2(F * G[None, :, :], s=(fh, fw), axes=(1, 2))
    return full[:, y0:y0+H, x0:x0+W]


def _e_fwd(A, D):
    return torch.einsum("khw,kt->hwt", A, D) if USE_TORCH else np.einsum("khw,kt->hwt", A, D)


def _e_bak(R, D):
    return torch.einsum("hwt,kt->khw", R, D) if USE_TORCH else np.einsum("hwt,kt->khw", R, D)


def _clip0(A):
    return torch.clamp(A, min=0.0) if USE_TORCH else np.clip(A, 0, None)


# ---------------------------------------------------------------- estimators (cloned)
def mle_gpu(Y_d, D_d, Dsum_d, bg, n_iter=N_ITER):
    K = D_d.shape[0]; H, W, T = Y_d.shape
    total = float(to_np(Y_d).sum())
    init = max(total / (H * W * K), 1e-4)
    A = (torch.full((K, H, W), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, H, W), init))
    for _ in range(n_iter):
        lam = _e_fwd(A, D_d) + bg
        R = Y_d / (lam + EPS)
        A = A * (_e_bak(R, D_d) / (Dsum_d.reshape(K, 1, 1) + EPS))
    return A


def joint_gpu(Y_d, D_d, o, C, bg, n_iter=N_ITER, eta=ETA):
    K = D_d.shape[0]; H, W, T = Y_d.shape
    total = float(to_np(Y_d).sum())
    init = max(total / (H * W * K), 1e-4)
    A = (torch.full((K, H, W), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, H, W), init))
    for _ in range(n_iter):
        blur = _fft_op(A, o)
        lam = _e_fwd(blur, D_d) + bg
        R = Y_d / (lam + EPS)
        U = _fft_op(_e_bak(R, D_d), o, adjoint=True)
        A = A * _clip0(U / (C + EPS)) ** eta
    return A


def rl_deconvolve(img, o, n_iter=N_ITER, eta=ETA):
    """RL control: the SAME multiplicative Poisson update, PSF and damping as
    the joint estimator, applied to one detector-space amplitude map."""
    x = np.clip(img, EPS, None)[None]              # (1,H,W)
    x = to_dev(x); y = to_dev(np.clip(img, 0, None)[None])
    ones = to_dev(np.ones_like(to_np(y)))
    C = _fft_op(ones, o)                            # PSF sensitivity
    for _ in range(n_iter):
        lam = _fft_op(x, o)
        R = y / (lam + EPS)
        U = _fft_op(R, o, adjoint=True)
        x = x * _clip0(U / (C + EPS)) ** eta
    return to_np(x)[0]


def lifetime_at(A, coords):
    return fs.lifetime_at_points(A, coords, np.array(fs.TAUS_NS), fallback=FALLBACK_TAU)


def photon_budget(Lam, coords):
    sig = Lam.sum(axis=2) - fs.BACKGROUND_RATE * Lam.shape[2]
    fg = sig > FG_FRAC * sig.max()
    ctr = float(np.mean([sig[y, x] for (x, y) in coords]))
    return float(sig[fg].mean()), ctr


# ---------------------------------------------------------------- one full pass
def run_pass(bg_mode, controls=False):
    """MLE + SPOOL (always); + Wiener/TV/RL when controls=True.
    Returns per-method rmse mean/std lists and the photon axis."""
    D_d = to_dev(fs.D_BASES)
    Dsum_d = to_dev(fs.D_BASES.sum(axis=1))
    o = make_otf(fs.PSF, (fs.IMAGE_SIZE, fs.IMAGE_SIZE))
    K = len(fs.TAUS_NS)
    ones = to_dev(np.ones((K, fs.IMAGE_SIZE, fs.IMAGE_SIZE)))
    Dsum_kd = Dsum_d.reshape(K, 1, 1)
    C = _fft_op(ones, o) * Dsum_kd

    methods = ["MLE", "Proposed"] + (["Wiener", "TV", "RL"] if controls else [])
    res = {m: {"rmse_mean": [], "rmse_std": []} for m in methods}
    ph_axis = []; bg_used = []

    for pl in PHOTON_LEVELS:
        fs.BACKGROUND_RATE = BG_FIXED if bg_mode == "fixed" else BG_SNB[pl]
        bg = fs.BACKGROUND_RATE
        per = {m: [] for m in methods}; fgm = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(2000 + trial)
            A_true, coords = fs.generate_scene(n_beads=N_BEADS, photons_per_bead=pl,
                                               rng=rng, n_species0=N_BEADS)  # homogeneous
            species = np.array([np.argmax(A_true[:, y, x]) for (x, y) in coords])
            tau_true = np.array(fs.TAUS_NS)[species]
            Y, Lam = fs.simulate_measurement(A_true, rng)
            f_, _ = photon_budget(Lam, coords); fgm.append(f_)

            Y_d = to_dev(Y)
            A_mle = to_np(mle_gpu(Y_d, D_d, Dsum_d, bg))
            per["MLE"].append(float(np.sqrt(np.mean(
                (lifetime_at(A_mle, coords) - tau_true) ** 2))))
            A_j = to_np(joint_gpu(Y_d, D_d, o, C, bg))
            per["Proposed"].append(float(np.sqrt(np.mean(
                (lifetime_at(A_j, coords) - tau_true) ** 2))))

            if controls:
                A_un = fs.pixelwise_nnls_unmix(Y)          # unmix ONCE
                A_w = np.stack([fs.wiener_deconvolve(A_un[k], fs.PSF, snr=50.0)
                                for k in range(K)])
                A_t = np.stack([fs.tv_deconvolve(A_un[k], fs.PSF, n_iter=100, lam=0.02)
                                for k in range(K)])
                A_r = np.stack([rl_deconvolve(A_un[k], o) for k in range(K)])
                for nm, Ax in (("Wiener", A_w), ("TV", A_t), ("RL", A_r)):
                    per[nm].append(float(np.sqrt(np.mean(
                        (lifetime_at(Ax, coords) - tau_true) ** 2))))
        for m in methods:
            res[m]["rmse_mean"].append(float(np.mean(per[m])))
            res[m]["rmse_std"].append(float(np.std(per[m])))
        ph_axis.append(float(np.mean(fgm))); bg_used.append(bg)
        line = f"  pl {pl:>4} ({ph_axis[-1]:5.2f} ph/px): " + "  ".join(
            f"{m} {res[m]['rmse_mean'][-1]:.3f}" for m in methods)
        print(line)
    return res, ph_axis, bg_used


def gate_check(res):
    dm = np.abs(np.array(res["MLE"]["rmse_mean"]) - GATE_MLE)
    dp = np.abs(np.array(res["Proposed"]["rmse_mean"]) - GATE_PROP)
    ok = bool(dm.max() < GATE_ATOL and dp.max() < GATE_ATOL)
    return ok, dm, dp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    patch_module()
    t0 = time.time()

    chosen = None
    for bg_mode in ("fixed", "snb"):
        print(f"\n=== gate pass: background mode '{bg_mode}' (MLE + SPOOL only) ===")
        res, ph, bgs = run_pass(bg_mode, controls=False)
        ok, dm, dp = gate_check(res)
        print(f"  gate: {'PASS' if ok else 'FAIL'}  "
              f"max|dMLE| {dm.max():.4f}  max|dSPOOL| {dp.max():.4f}")
        if ok:
            chosen = bg_mode
            break
    if chosen is None:
        print("\nNEITHER background mode reproduces the main benchmark.")
        print("Per-level differences (fixed-mode shown above, snb-mode below);")
        print("check joint/MLE iteration count and the seed schedule of the")
        print("run that produced the main it50 results.json. Nothing rendered.")
        return

    print(f"\n=== full run with controls (background mode '{chosen}') ===")
    res, ph, bgs = run_pass(chosen, controls=True)
    ok, dm, dp = gate_check(res)
    assert ok, "gate regressed on the full run -- investigate before using output"

    M = np.array(res["MLE"]["rmse_mean"]); P = np.array(res["Proposed"]["rmse_mean"])
    W_ = np.array(res["Wiener"]["rmse_mean"]); T_ = np.array(res["TV"]["rmse_mean"])
    R_ = np.array(res["RL"]["rmse_mean"])
    best = np.minimum(np.minimum(W_, T_), R_)
    ratio_best = best / P
    rl_frac = (M - R_) / (M - P)

    # ---------------- figure ----------------
    colors = {"MLE": "#888888", "Wiener": "#0072B2", "TV": "#E69F00",
              "RL": "#009E73", "SPOOL": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for nm, key, mk in (("MLE", "MLE", "o"), ("Wiener", "Wiener", "s"),
                        ("TV", "TV", "^"), ("RL", "RL", "v"),
                        ("SPOOL", "Proposed", "D")):
        ax.errorbar(ph, res[key]["rmse_mean"], yerr=res[key]["rmse_std"],
                    fmt=f"{mk}-", color=colors[nm], lw=2, ms=7, capsize=3, label=nm)
    ax.set_xscale("log")
    ax.set_xticks(ph); ax.set_xticklabels([f"{v:.1f}" for v in ph])
    ax.minorticks_off()
    ax.set_xlabel("Mean photons per foreground pixel", fontsize=13)
    ax.set_ylabel("Lifetime RMSE (ns)", fontsize=13)
    ax.tick_params(labelsize=11); ax.legend(fontsize=10, frameon=False)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Supple_1.png"), dpi=220)
    plt.close(fig)

    # ---------------- serialize ----------------
    payload = dict(background_mode=chosen, bg_per_bin=bgs,
                   photons_fg_mean=ph, tau_generation=TAU_GEN,
                   n_iter=N_ITER, eta=ETA, fallback=FALLBACK_TAU,
                   gate="PASS",
                   ratio_best_sequential=[float(v) for v in ratio_best],
                   rl_gap_fraction=[float(v) for v in rl_frac],
                   **{k: v for k, v in res.items()})
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(payload, f, indent=1)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Sequential controls at {N_ITER} iterations, background '{chosen}', "
                f"gate PASS vs main it50 benchmark\n")
        f.write(f"{'ph/px':>7}{'MLE':>8}{'Wiener':>8}{'TV':>8}{'RL':>8}{'SPOOL':>8}"
                f"{'best/SP':>9}{'RLfrac':>8}\n")
        for i in range(len(ph)):
            f.write(f"{ph[i]:>7.1f}{M[i]:>8.3f}{W_[i]:>8.3f}{T_[i]:>8.3f}{R_[i]:>8.3f}"
                    f"{P[i]:>8.3f}{ratio_best[i]:>9.2f}{100*rl_frac[i]:>7.0f}%\n")
        f.write(f"\nbest-sequential / SPOOL: {ratio_best.min():.2f}-{ratio_best.max():.2f}\n")
        f.write(f"RL closes at most {100*rl_frac.max():.0f}% of the MLE->SPOOL distance\n")
        bn = ["Wiener", "TV", "RL"]
        f.write("best sequential per level: "
                + ", ".join(bn[int(i)] for i in np.argmin(np.vstack([W_, T_, R_]), axis=0)) + "\n")
    print(f"\n({time.time()-t0:.0f}s) -> {OUT_DIR}: Supple_1.png, results.json, summary.txt")


if __name__ == "__main__":
    main()

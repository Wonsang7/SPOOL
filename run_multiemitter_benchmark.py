"""
Simulation benchmark on bead phantoms, for TWO scene types, GPU-accelerated.

  heterogeneous : beads split between tau = 2 ns and tau = 4 ns
  homogeneous   : every bead has tau = 2 ns

THREE THINGS THIS VERSION GETS RIGHT
------------------------------------
1. LIFETIME-AGNOSTIC RECONSTRUCTION DICTIONARY. Earlier versions reconstructed
   the phantoms with the two TRUE decay bases. To keep reconstruction agnostic
   and matched to the experimental analysis, generation and reconstruction are
   separated:

       GENERATION      K = 2,  tau = {2, 4} ns          (the truth)
       RECONSTRUCTION  K = 11, tau = 1.0-5.0 ns, 0.4    (as in the experiments)

   Every dictionary-based method -- pixel-wise MLE, NC-PCA + NNLS and the
   proposed estimator -- sees the same agnostic dictionary used on the real
   data. Phasor is dictionary-free and unaffected.

2. FIXED DETECTION-FAILURE COST. fs.lifetime_at_points normally falls back to
   the mean of whatever dictionary it is given when a method recovers no
   amplitude at a point. FALLBACK_TAU pins that value to 3.0 ns, the midpoint
   of the true 2- and 4-ns lifetimes, so the cost of a detection failure does
   not change if the reconstruction dictionary is modified.

3. BACKGROUND AT A CONSTANT SIGNAL-TO-BACKGROUND RATIO. A fixed background
   makes the ratio vary sixteenfold across the photon sweep, which is not what a
   real acquisition does: binomial thinning removes background counts with the
   same probability as signal counts, so the ratio is preserved. The background
   is therefore recalibrated at each photon level to hold SNB fixed, matching
   the experimental protocol. SNB = 50 is far more conservative than the
   measurement: the 114 pre-pulse bins of the microtubule acquisition contain no
   counts at all across the whole field, bounding its background below
   2.4e-6 counts per pixel per bin, i.e. SNB above 10^5.

REQUIRES: flim_sim_ncpca.py in the same folder.

    python run_multiemitter_benchmark.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time
import contextlib
import numpy as np
from scipy.special import erf, erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors, matplotlib.cm, matplotlib.ticker

import flim_sim_ncpca as fs

# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("SPOOL_RESULTS_DIR", os.path.join(HERE, "results", "multiemitter"))

SCENES = {
    "heterogeneous": dict(n_species0=None),
    "homogeneous":   dict(n_species0=15),
}

N_BEADS        = 15
PHOTON_LEVELS  = [50, 100, 200, 400, 800]   # photons per bead
QUAL_PHOTONS   = 200
N_TRIALS       = 8
N_ITER         = 50
ETA            = 0.9
PCA_COMPONENTS = 3
PHASOR_MHZ     = 80.0
PHASOR_RANGE   = (0.0, 10.0)
LIFE_LO, LIFE_HI = 2, 4
EPS = 1e-9

FG_FRAC      = 0.15        # foreground = pixels above this fraction of the peak
FALLBACK_TAU = 3.0         # midpoint of the TRUE lifetimes; see the header

# --- reconstruction dictionary: lifetime-agnostic, as in the experiments -----
TAU_REC_MIN, TAU_REC_MAX, TAU_REC_STEP = 1.0, 5.0, 0.4      # -> K = 11

# --- forward model, matched to single_bead_crb_mc.py ------------------------
MATCH_SINGLE_EMITTER = True
PIXEL_NM      = 50
PSF_FWHM_NM   = 200.0
PSF_INTEGRATE = True

# --- background -------------------------------------------------------------
# "snb"   : recalibrated at each photon level so that the ratio of the mean
#           foreground signal to the total background is constant, as it is
#           under binomial thinning of a real acquisition.
# "fixed" : constant BG_PER_BIN regardless of photon level (the earlier
#           behaviour; kept so the two can be compared).
BACKGROUND_MODE = "snb"
SNB             = 50      # mean foreground signal / background, per pixel
BG_PER_BIN      = 1e-4      # used only when BACKGROUND_MODE == "fixed"
# ===============================================================

# ---------------------------------------------------------------- backend
USE_TORCH = False
DEVICE = None
try:
    import torch
    if torch.cuda.is_available():
        USE_TORCH = True
        DEVICE = torch.device("cuda")
        BACKEND = f"PyTorch / GPU ({torch.cuda.get_device_name(0)})"
    else:
        BACKEND = "NumPy / CPU (torch present, CUDA unavailable)"
except ImportError:
    BACKEND = "NumPy / CPU (torch not installed)"


def to_dev(a):
    if USE_TORCH:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=DEVICE)
    return np.asarray(a, dtype=np.float64)


def to_np(a):
    if USE_TORCH and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


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


def match_forward_model():
    sigma_px = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM
    psf_new = gaussian_psf(sigma_px, PSF_INTEGRATE)
    old_h0, old_bg = float(np.asarray(fs.PSF).max()), fs.BACKGROUND_RATE
    fs.PSF = psf_new
    fs.BACKGROUND_RATE = BG_PER_BIN          # overridden per level in "snb" mode
    nT = int(getattr(fs, "N_TIME_BINS", 256))
    print("\nForward model matched to single_bead_crb_mc.py:")
    print(f"  PSF   : sigma {sigma_px:.3f} px, {psf_new.shape[0]}x{psf_new.shape[1]} kernel, "
          f"{'pixel-integrated' if PSF_INTEGRATE else 'point-sampled'}")
    print(f"          h(0) {old_h0:.5f} -> {psf_new.max():.5f}   "
          f"predicted gain {np.sqrt(1/psf_new.max()):.2f}x")
    if BACKGROUND_MODE == "snb":
        print(f"  bkgnd : constant signal-to-background ratio, SNB = {SNB:g};"
              f" recalibrated at each photon level")
    else:
        print(f"  bkgnd : {old_bg:.3g} -> {BG_PER_BIN:.3g} per pixel per bin "
              f"({old_bg*nT:.4f} -> {BG_PER_BIN*nT:.4f} photons per pixel)")


# ------------------------------------------------- reconstruction dictionary
def build_decay_basis(taus):
    """IRF-convolved, window-normalized exponential decays, Eq. (S2)."""
    N_T = int(fs.N_TIME_BINS)
    dt = float(fs.DT_NS)
    t = np.arange(N_T) * dt
    t0 = float(fs.IRF_PEAK_BIN) * dt
    sig = float(fs.IRF_FWHM_NS) / 2.3548200450309493
    tau = np.atleast_1d(np.asarray(taus, dtype=np.float64))[:, None]
    tt = t[None, :]
    lam = 1.0 / tau
    arg = (t0 + lam * sig ** 2 - tt) / (np.sqrt(2) * sig)
    ex = np.clip(0.5 * lam ** 2 * sig ** 2 - lam * (tt - t0), -700, 700)
    d = np.clip(0.5 * lam * np.exp(ex) * erfc(arg), 0.0, None)
    s = d.sum(axis=1, keepdims=True)
    return d / np.where(s > 0, s, 1.0)


TAU_REC = np.round(np.arange(TAU_REC_MIN, TAU_REC_MAX + 1e-9, TAU_REC_STEP), 4)
K_REC = len(TAU_REC)
D_REC = None            # built in main(), after fs has been configured


@contextlib.contextmanager
def reconstruction_dictionary():
    """Swap the module's dictionary globals for the agnostic one.

    fs.ncpca_denoise_unmix reads D_BASES, TAUS_NS and K_SPECIES from module
    scope, so the swap has to be visible to it. Scene generation and measurement
    simulation must NOT run inside this block: they define the ground truth and
    use the two true bases.
    """
    keep = (fs.D_BASES, fs.TAUS_NS, fs.K_SPECIES)
    fs.D_BASES, fs.TAUS_NS, fs.K_SPECIES = D_REC, TAU_REC, K_REC
    try:
        yield
    finally:
        fs.D_BASES, fs.TAUS_NS, fs.K_SPECIES = keep


# ---------------------------------------------------------------- FFT ops
def make_otf(psf, shape):
    H, W = shape
    ph, pw = psf.shape
    fh, fw = H + ph - 1, W + pw - 1
    pad = np.zeros((fh, fw)); pad[:ph, :pw] = psf
    pad_f = np.zeros((fh, fw)); pad_f[:ph, :pw] = psf[::-1, ::-1]
    if USE_TORCH:
        otf = torch.fft.rfft2(to_dev(pad), s=(fh, fw))
        otf_adj = torch.fft.rfft2(to_dev(pad_f), s=(fh, fw))
    else:
        otf = np.fft.rfft2(pad, s=(fh, fw))
        otf_adj = np.fft.rfft2(pad_f, s=(fh, fw))
    y0, x0 = (ph - 1) // 2, (pw - 1) // 2
    return dict(otf=otf, otf_adj=otf_adj, fshape=(fh, fw), out=(H, W), off=(y0, x0))


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


def conv_batch(A, o): return _fft_op(A, o, adjoint=False)
def corr_batch(A, o): return _fft_op(A, o, adjoint=True)


def _einsum_khw_kt(A, D):
    return torch.einsum("khw,kt->hwt", A, D) if USE_TORCH else np.einsum("khw,kt->hwt", A, D)


def _einsum_hwt_kt(R, D):
    return torch.einsum("hwt,kt->khw", R, D) if USE_TORCH else np.einsum("hwt,kt->khw", R, D)


def _clip0(A):
    return torch.clamp(A, min=0.0) if USE_TORCH else np.clip(A, 0, None)


# ---------------------------------------------------------------- recon
def mle_gpu(Y_d, D_d, Dsum_d, bg, n_iter=N_ITER):
    K = D_d.shape[0]; H, W, T = Y_d.shape
    init = max(float(to_np(Y_d).sum()) / (H * W * K), 1e-4)
    A = (torch.full((K, H, W), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, H, W), init))
    for _ in range(n_iter):
        lam = _einsum_khw_kt(A, D_d) + bg
        A = A * (_einsum_hwt_kt(Y_d / (lam + EPS), D_d) / (Dsum_d.reshape(K, 1, 1) + EPS))
    return A


def joint_gpu(Y_d, D_d, o, C, bg, n_iter=N_ITER, eta=ETA):
    K = D_d.shape[0]; H, W, T = Y_d.shape
    init = max(float(to_np(Y_d).sum()) / (H * W * K), 1e-4)
    A = (torch.full((K, H, W), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, H, W), init))
    for _ in range(n_iter):
        lam = _einsum_khw_kt(conv_batch(A, o), D_d) + bg
        U = corr_batch(_einsum_hwt_kt(Y_d / (lam + EPS), D_d), o)
        A = A * _clip0(U / (C + EPS)) ** eta
        A = torch.clamp(A, min=1e-8) if USE_TORCH else np.maximum(A, 1e-8)
    return A


# ---------------------------------------------------------------- helpers
def tau_map(A, taus):
    tot = A.sum(0)
    return np.divide((np.asarray(taus)[:, None, None] * A).sum(0), tot,
                     out=np.zeros_like(tot), where=tot > EPS)


def lifetime_at(A, coords, taus):
    return fs.lifetime_at_points(A, coords, taus, fallback=FALLBACK_TAU)


def background_for_level(pl, cfg):
    """Background rate giving the configured signal-to-background ratio.

    The foreground mean is measured on a noiseless, background-free realization
    of the same scene, so the calibration carries no shot noise and depends only
    on the photon level. Under binomial thinning a real acquisition preserves
    this ratio, which a fixed background does not.
    """
    if BACKGROUND_MODE != "snb":
        return BG_PER_BIN
    keep = fs.BACKGROUND_RATE
    fs.BACKGROUND_RATE = 0.0
    try:
        A0, _ = fs.generate_scene(n_beads=N_BEADS, photons_per_bead=pl,
                                  rng=np.random.default_rng(2000),
                                  n_species0=cfg["n_species0"])
        inten = fs.forward_model(A0).sum(axis=2)
    finally:
        fs.BACKGROUND_RATE = keep
    fg = inten[inten >= FG_FRAC * float(inten.max())].mean()
    return float(fg / SNB / int(fs.N_TIME_BINS))


def photon_budget(Lam, coords):
    inten = Lam.sum(axis=2)
    fg = inten >= FG_FRAC * float(inten.max())
    centre = np.array([inten[y, x] for (x, y) in coords]).mean()
    return float(inten[fg].mean()), float(centre)


def masked_map(tau, weight, frac=0.1):
    thr = frac * max(float(weight.max()), EPS)
    return np.ma.array(tau, mask=(weight < thr) | ~np.isfinite(tau) | (tau <= 0))


def save_panel(img, path, cmap, vmin, vmax):
    h, w = img.shape; dpi = 200
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.savefig(path, dpi=dpi); plt.close(fig)
    print(f"    saved {os.path.basename(path)}")


def save_colorbar(cmap, vmin, vmax, label, path):
    fig, ax = plt.subplots(figsize=(1.3, 4.5))
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=ax)
    cb.set_label(label, fontsize=13); cb.ax.tick_params(labelsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def eval_phasor(Y, coords, tau_true):
    th = fs.phasor_lifetime_at_points(Y, coords, dt_ns=fs.DT_NS, freq_mhz=PHASOR_MHZ)
    lo, hi = PHASOR_RANGE
    ok = np.isfinite(th) & (th > lo) & (th < hi)
    if ok.sum() == 0:
        return np.nan, 100.0
    return float(np.sqrt(np.mean((th[ok] - tau_true[ok])**2))), 100.0*(1 - ok.sum()/len(th))


# ---------------------------------------------------------------- main
def run_scene(name, cfg):
    print(f"\n=== {name} ===")
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    cmap = plt.cm.rainbow.copy(); cmap.set_bad("black")

    TAU_GEN = np.asarray(fs.TAUS_NS)          # the two true bases

    D_d = to_dev(D_REC)
    Dsum_d = to_dev(D_REC.sum(axis=1))
    H = W = fs.IMAGE_SIZE
    o = make_otf(np.asarray(fs.PSF), (H, W))
    ones = (torch.ones((K_REC, H, W), dtype=torch.float64, device=DEVICE)
            if USE_TORCH else np.ones((K_REC, H, W)))
    C = conv_batch(ones, o) * Dsum_d.reshape(K_REC, 1, 1)

    methods = ["MLE", "PCA", "Phasor", "Proposed"]
    res = {m: {"rmse_mean": [], "rmse_std": []} for m in methods}
    res["photons_fg_mean"] = []; res["photons_centre_px"] = []
    res["emitter_photons"] = []; res["phasor_failure_pct"] = []
    res["photons_per_bead"] = list(PHOTON_LEVELS)
    res["background_mode"] = BACKGROUND_MODE
    res["snb"] = SNB if BACKGROUND_MODE == "snb" else None
    res["bg_per_bin"] = []
    res["tau_reconstruction"] = TAU_REC.tolist()
    res["tau_generation"] = TAU_GEN.tolist()
    qual = None

    for pl in PHOTON_LEVELS:
        # set the background for this level, then use the SAME value for
        # generation and for every reconstruction
        bg = background_for_level(pl, cfg)
        fs.BACKGROUND_RATE = bg
        per = {m: [] for m in methods}; fails = []; fgm = []; ctr = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(2000 + trial)

            # ---- generation: the TRUE two-basis model defines the ground truth
            A_true, coords = fs.generate_scene(n_beads=N_BEADS, photons_per_bead=pl,
                                               rng=rng, n_species0=cfg["n_species0"])
            species = np.array([np.argmax(A_true[:, y, x]) for (x, y) in coords])
            tau_true = TAU_GEN[species]
            Y, Lam = fs.simulate_measurement(A_true, rng)
            f_, c_ = photon_budget(Lam, coords)
            fgm.append(f_); ctr.append(c_)

            # ---- reconstruction: every method sees the agnostic K=11 dictionary
            with reconstruction_dictionary():
                Y_d = to_dev(Y)
                A_mle = to_np(mle_gpu(Y_d, D_d, Dsum_d, bg))
                per["MLE"].append(float(np.sqrt(np.mean(
                    (lifetime_at(A_mle, coords, TAU_REC) - tau_true) ** 2))))

                A_pca = fs.ncpca_denoise_unmix(Y, n_components=PCA_COMPONENTS)
                per["PCA"].append(float(np.sqrt(np.mean(
                    (lifetime_at(A_pca, coords, TAU_REC) - tau_true) ** 2))))

                r_ph, fail = eval_phasor(Y, coords, tau_true)   # dictionary-free
                per["Phasor"].append(r_ph); fails.append(fail)

                A_j = to_np(joint_gpu(Y_d, D_d, o, C, bg))
                per["Proposed"].append(float(np.sqrt(np.mean(
                    (lifetime_at(A_j, coords, TAU_REC) - tau_true) ** 2))))

            if pl == QUAL_PHOTONS and trial == 0:
                qual = dict(A_true=A_true, Y=Y, A_mle=A_mle, A_pca=A_pca, A_joint=A_j)

        for m in methods:
            res[m]["rmse_mean"].append(float(np.nanmean(per[m])))
            res[m]["rmse_std"].append(float(np.nanstd(per[m])))
        res["photons_fg_mean"].append(float(np.mean(fgm)))
        res["photons_centre_px"].append(float(np.mean(ctr)))
        res["emitter_photons"].append(float(np.mean(fgm)))
        res["phasor_failure_pct"].append(float(np.mean(fails)))
        res["bg_per_bin"].append(bg)
        print(f"  {pl:>4} ph/bead -> {np.mean(fgm):5.2f} ph/px foreground "
              f"({np.mean(ctr):5.2f} centre, bg {bg*fs.N_TIME_BINS:.4f} ph/px): " +
              "  ".join(f"{m} {np.nanmean(per[m]):.3f}" for m in methods))

    print("  qualitative panels:")
    save_panel(masked_map(tau_map(qual["A_true"], TAU_GEN), qual["A_true"].sum(0), frac=0.5),
               os.path.join(out, "qual_GroundTruth.png"), cmap, LIFE_LO, LIFE_HI)
    save_panel(masked_map(tau_map(qual["A_mle"], TAU_REC), qual["A_mle"].sum(0)),
               os.path.join(out, "qual_MLE.png"), cmap, LIFE_LO, LIFE_HI)
    save_panel(masked_map(tau_map(qual["A_pca"], TAU_REC), qual["A_pca"].sum(0)),
               os.path.join(out, "qual_PCA.png"), cmap, LIFE_LO, LIFE_HI)
    tau_ph, _, _ = fs.phasor_lifetime_map(qual["Y"], dt_ns=fs.DT_NS, freq_mhz=PHASOR_MHZ)
    save_panel(masked_map(tau_ph, qual["Y"].sum(axis=2)),
               os.path.join(out, "qual_Phasor.png"), cmap, LIFE_LO, LIFE_HI)
    save_panel(masked_map(tau_map(qual["A_joint"], TAU_REC), qual["A_joint"].sum(0)),
               os.path.join(out, "qual_Proposed.png"), cmap, LIFE_LO, LIFE_HI)
    save_colorbar(cmap, LIFE_LO, LIFE_HI, "Lifetime (ns)",
                  os.path.join(out, "colorbar_lifetime.png"))

    style = {"MLE": ("#7f7f7f", "o"), "PCA": ("#1f77b4", "s"),
             "Phasor": ("#9467bd", "v"), "Proposed": ("#d62728", "D")}
    fig, ax = plt.subplots(figsize=(7, 6))
    x = res["photons_fg_mean"]
    for m in methods:
        c, mk = style[m]
        ax.errorbar(x, res[m]["rmse_mean"], yerr=res[m]["rmse_std"],
                    color=c, marker=mk, lw=2, ms=8, capsize=3, label=m)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in x])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Mean photons per foreground pixel", fontsize=15)
    ax.set_ylabel("Lifetime RMSE (ns)", fontsize=15)
    ax.tick_params(labelsize=13); ax.grid(alpha=0.25); ax.legend(fontsize=12, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "sweep_rmse.png"), dpi=180)
    plt.close(fig); print("    saved sweep_rmse.png")

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


def main():
    global D_REC
    print(f"Backend: {BACKEND}")
    if MATCH_SINGLE_EMITTER:
        match_forward_model()

    D_REC = build_decay_basis(TAU_REC)

    print("\nDictionaries:")
    print(f"  generation     : K = {fs.K_SPECIES}, tau = {np.asarray(fs.TAUS_NS)} ns "
          f"(defines the ground truth)")
    print(f"  reconstruction : K = {K_REC}, tau = {TAU_REC[0]}-{TAU_REC[-1]} ns "
          f"step {TAU_REC_STEP} (lifetime-agnostic, as in the experiments)")
    print(f"  detection-failure fallback : {FALLBACK_TAU} ns "
          f"(midpoint of the true lifetimes, not of the dictionary)")
    on_grid = [float(t) for t in np.asarray(fs.TAUS_NS)
               if np.min(np.abs(TAU_REC - t)) < 1e-6]
    if on_grid:
        print(f"  note: true lifetimes {on_grid} ns lie exactly on reconstruction")
        print(f"        nodes, so quantization bias is zero for them; the")
        print(f"        single-emitter study places its truth midway on purpose.")

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    for name, cfg in SCENES.items():
        run_scene(name, cfg)
    print(f"\nAll done in {time.time()-t0:.0f}s. Panels in {OUT_DIR}")


if __name__ == "__main__":
    main()
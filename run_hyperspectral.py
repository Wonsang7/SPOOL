"""
Hyperspectral photon efficiency: the same estimator, with the decay dictionary
swapped for a spectral one.

The point of this experiment is not spectral unmixing per se. It is to test
whether the mechanism claimed for FLIM -- that a pixel-wise estimator discards
photon information the PSF has delocalized, and that an explicit optical forward
model recovers it -- is really about photons and optics rather than about
lifetime. If it is, then replacing D_k(t) with S_k(lambda) and changing nothing
else should reproduce the same behavior on a completely different physical
quantity.

Forward model (identical in structure to the FLIM one):

    Lambda(r, lam) = sum_k [h * A_k](r) * S_k(lam) + B

The spectral dictionary S_k is MODEL-FREE, exactly as the decay dictionary is:
a bank of Gaussian emission bands tiling the detected wavelength range. No
fluorophore spectrum is assumed, just as no dye lifetime is assumed in the FLIM
case. The recovered quantity is the abundance-weighted mean emission wavelength,

    lambda_hat(r) = sum_k mu_k A_k(r) / sum_k A_k(r),

which is the exact analogue of the abundance-weighted mean lifetime.

The 4D acquisition (Y, X, T, lambda) is collapsed over the decay axis: the time
dimension is used only to give a high-photon starting point that can be thinned,
not as an estimation channel. This keeps the comparison clean -- the spectral
model has access to exactly one physical channel, as the FLIM model does.

Baselines:
  - pixel-wise Poisson MLE : the same estimator with h -> delta. This isolates
                             the spatial coupling, which is the claim under test.
  - pixel-wise NNLS        : the standard practical unmixing, which assumes
                             Gaussian noise and is included to show what that
                             assumption costs at low photon counts.

Metric: RMSE of the recovered mean emission wavelength against each method's own
full-photon reconstruction, with the bias and standard-deviation parts reported
separately -- they answer different questions, and reporting only their
combination hides which one is moving. Each method is scored and displayed only
where it recovered abundance above ABUND_FRAC of its own maximum, so the two are
evaluated on different pixel counts by construction; those counts are written to
the summary rather than suppressed.

USAGE:
    python hyperspectral_panels.py
"""

import os

# Windows/Anaconda: torch and NumPy(MKL) can each load their own Intel OpenMP
# runtime, which aborts with "OMP: Error #15". Must be set before the imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import numpy as np
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors, matplotlib.cm

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


# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get(
    "SPOOL_HYPERSPECTRAL_RAW",
    os.path.join(HERE, "data", "hyperspectral", "decay_YXB_lambda.npy"),
)
OUT_DIR = os.environ.get(
    "SPOOL_HYPERSPECTRAL_RESULTS_DIR",
    os.path.join(HERE, "results", "hyperspectral"),
)

# Axis order of the file, as identified by inspect_4d.py:
#   axis 2 is the DECAY axis (skew +1.98), axis 3 is the SPECTRAL axis.
Y_AX, X_AX, T_AX, LAM_AX = 0, 1, 2, 3

# Spectral calibration (from the acquisition folder name: 520 to 650, step 5)
LAMBDA_START_NM = 520.0
LAMBDA_STEP_NM  = 5.0

CROP_SIZE   = 512          # centre crop from the 1024x1024 field
CROP_CORNER = "center"

# --- optics (STELLARIS confocal, HC PL APO CS2 63x/1.40 OIL, 1 AU pinhole) ---
PIXEL_NM = 50.0            # from the acquisition metadata
NA       = 1.40            # objective numerical aperture

# Manuscript-wide nominal PSF rule: FWHM = 0.51 * lambda / NA, evaluated at a
# representative emission wavelength of 550 nm and rounded to 200 nm. A single,
# spatially invariant, wavelength-independent PSF is used in every reconstruction.
PSF_LAMBDA_NM = 550.0
PSF_FWHM_NM   = 200.0

# --- model-free spectral dictionary (the analogue of the decay dictionary) ---
N_BASES      = 11          # number of Gaussian emission bands tiling the range
BASIS_FWHM_NM = 40.0       # width of each band; typical fluorophore emission FWHM

# --- display ---
# The recovered quantity is an abundance-weighted mean emission wavelength, which
# is a mixture coordinate rather than a pure-dye wavelength: within one confocal
# volume both fluorophores contribute, so almost every pixel lies between the two
# component signatures. The colour range therefore has to span the mixture range,
# not the dyes' emission maxima.
#   None            -> take it from the data, at the percentiles in COLOR_PCT
#   (lo, hi) in nm  -> fix it, e.g. (555.0, 595.0)
WAVELENGTH_RANGE_NM = (560.0, 585.0)
COLOR_PCT = (0.5, 99.5)    # used only when WAVELENGTH_RANGE_NM is None

# |error| colour scale, shared by every error map so they are directly comparable
ERROR_RANGE_NM = (0.0, 20.0)

# Scale bar, drawn on the reference panels. 50 nm pixels over a 512 px crop give
# a 25.6 um field, so 5 um is 100 px -- a fifth of the image width.
SCALEBAR_UM = 5.0

# --- photon budgets ---
SWEEP_PHOTONS   = [1, 3, 5, 7, 9, 11, 13, 15]
DISPLAY_PHOTONS = [1, 5]
N_TRIALS        = 6
N_ITER          = 50       # same as the FLIM experiments
ETA             = 0.9
ABUND_FRAC      = 0.03     # each method masked by its own recovered abundance
FG_THRESHOLD_FRAC = 0.15
EPS = 1e-9
# ===============================================================


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- FFT ops
# Zero-padded linear convolution, matching scipy's fftconvolve(mode="same"):
# a plain FFT product would wrap the opposite image edge around.
def make_otf(psf, shape):
    H, W = shape; ph, pw = psf.shape
    fh, fw = H + ph - 1, W + pw - 1
    pad = np.zeros((fh, fw)); pad[:ph, :pw] = psf
    pad_f = np.zeros((fh, fw)); pad_f[:ph, :pw] = psf[::-1, ::-1]
    if USE_TORCH:
        otf = torch.fft.rfft2(to_dev(pad), s=(fh, fw))
        otf_adj = torch.fft.rfft2(to_dev(pad_f), s=(fh, fw))
    else:
        otf = np.fft.rfft2(pad, s=(fh, fw))
        otf_adj = np.fft.rfft2(pad_f, s=(fh, fw))
    y0, x0 = (ph - 1)//2, (pw - 1)//2
    return dict(otf=otf, otf_adj=otf_adj, fshape=(fh, fw), out=(H, W), off=(y0, x0))


def fft_op(A, o, adjoint=False):
    fh, fw = o["fshape"]; H, W = o["out"]; y0, x0 = o["off"]
    G = o["otf_adj"] if adjoint else o["otf"]
    if USE_TORCH:
        F = torch.fft.rfft2(A, s=(fh, fw))
        full = torch.fft.irfft2(F * G.unsqueeze(0), s=(fh, fw))
    else:
        F = np.fft.rfft2(A, s=(fh, fw), axes=(1, 2))
        full = np.fft.irfft2(F * G[None, :, :], s=(fh, fw), axes=(1, 2))
    return full[:, y0:y0+H, x0:x0+W]


def es_khw_kt(A, S):
    return torch.einsum("khw,kl->hwl", A, S) if USE_TORCH else np.einsum("khw,kl->hwl", A, S)


def es_hwt_kt(R, S):
    return torch.einsum("hwl,kl->khw", R, S) if USE_TORCH else np.einsum("hwl,kl->khw", R, S)


# ---------------------------------------------------------------- estimators
def pixelwise_mle(Y, S, Ssum, bg, n_iter=N_ITER):
    """Pixel-wise Poisson MLE: the same estimator with h -> delta. No spatial
    coupling, which is exactly the term whose contribution we are measuring."""
    Yd = to_dev(Y); K = S.shape[0]; H, W, L = Y.shape
    init = max(float(np.asarray(Y).sum())/(H*W*K), 1e-4)
    A = (torch.full((K, H, W), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, H, W), init))
    Sd = to_dev(S); Ssd = to_dev(Ssum).reshape(K, 1, 1)
    for _ in range(n_iter):
        lam = es_khw_kt(A, Sd) + bg
        R = Yd/(lam + EPS)
        A = A * (es_hwt_kt(R, Sd) / (Ssd + EPS))
    return to_np(A)


def joint_recon(Y, S, Ssum, o, C, bg, n_iter=N_ITER, eta=ETA):
    """Joint spatial(PSF) + spectral Poisson reconstruction -- structurally
    identical to the FLIM estimator, with S_k(lambda) in place of D_k(t)."""
    Yd = to_dev(Y); K = S.shape[0]; H, W, L = Y.shape
    init = max(float(np.asarray(Y).sum())/(H*W*K), 1e-4)
    A = (torch.full((K, H, W), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, H, W), init))
    Sd = to_dev(S)
    for _ in range(n_iter):
        blur = fft_op(A, o)
        lam = es_khw_kt(blur, Sd) + bg
        R = Yd/(lam + EPS)
        U = fft_op(es_hwt_kt(R, Sd), o, adjoint=True)
        ratio = U/(C + EPS)
        ratio = torch.clamp(ratio, min=0.0) if USE_TORCH else np.clip(ratio, 0, None)
        A = A * ratio**eta
        A = torch.clamp(A, min=1e-8) if USE_TORCH else np.maximum(A, 1e-8)
    return to_np(A)


def pixelwise_nnls(Y, S):
    """The standard practical unmixing: least squares per pixel. Assumes Gaussian
    noise, which is exactly wrong at these photon counts -- included to show the
    cost of that assumption rather than as a straw man."""
    H, W, L = Y.shape; K = S.shape[0]
    flat = Y.reshape(H*W, L)
    A = np.zeros((K, H*W))
    B = S.T                                   # (L, K)
    for i in range(H*W):
        A[:, i], _ = nnls(B, flat[i])
    return A.reshape(K, H, W)


def mean_wavelength(A, mu):
    tot = A.sum(0)
    return np.divide((mu[:, None, None]*A).sum(0), tot,
                     out=np.zeros_like(tot), where=tot > EPS)


# ---------------------------------------------------------------- setup
def build_spectral_dictionary(L):
    """Model-free spectral dictionary: Gaussian emission bands tiling the
    detected range. No fluorophore spectrum is assumed -- the direct analogue of
    the lifetime-agnostic decay dictionary used in the FLIM experiments."""
    lam = LAMBDA_START_NM + np.arange(L)*LAMBDA_STEP_NM
    lo, hi = lam[0], lam[-1]
    mu = np.linspace(lo + 0.1*(hi-lo), hi - 0.1*(hi-lo), N_BASES)
    sigma = BASIS_FWHM_NM / 2.3548200450309493
    S = np.exp(-((lam[None, :] - mu[:, None])**2) / (2*sigma**2))
    S = S / S.sum(axis=1, keepdims=True)      # unit area, as the decay bases are
    log(f"  spectral dictionary: {N_BASES} bands, {mu[0]:.0f}-{mu[-1]:.0f} nm, "
        f"FWHM {BASIS_FWHM_NM:.0f} nm")
    return lam, mu, S


def estimate_psf_sigma_px():
    """PSF width in pixels. If the pixel size is unknown we cannot convert the
    optical FWHM into pixels, so this must be supplied -- it is the single most
    sensitive input to the reconstruction and should not be guessed silently."""
    if PIXEL_NM is None:
        raise SystemExit("PIXEL_NM must be set; the PSF cannot be inferred from the array.")
    sigma_px = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM
    log(f"  PSF: 0.51 * {PSF_LAMBDA_NM:.0f} nm / NA {NA:.2f} = {PSF_FWHM_NM:.0f} nm FWHM")
    log(f"       at {PIXEL_NM:.1f} nm/px -> sigma = {sigma_px:.2f} px")
    r = int(np.ceil(4*sigma_px)); a = np.arange(-r, r+1)
    xx, yy = np.meshgrid(a, a)
    h = np.exp(-(xx**2+yy**2)/(2*sigma_px**2)); h /= h.sum()
    n_eff = 1.0/np.sum(h**2)
    log(f"       n_eff = {n_eff:.1f} pixels -> heuristic gain scale "
        f"sqrt(n_eff) = {np.sqrt(n_eff):.1f}x")
    return sigma_px


def make_psf(sigma_px):
    r = int(np.ceil(4*sigma_px)); a = np.arange(-r, r+1)
    xx, yy = np.meshgrid(a, a)
    h = np.exp(-(xx**2 + yy**2)/(2*sigma_px**2))
    return h / h.sum()


# If make_thinning_stack.py has already collapsed and cropped the 7 GB file,
# reuse its output rather than re-reading the whole acquisition.
COLLAPSED_PATH = os.environ.get(
    "SPOOL_HYPERSPECTRAL_CUBE",
    os.path.join(HERE, "data", "hyperspectral", "hyperspectral_full.npy"),
)


def load_cube():
    if COLLAPSED_PATH and os.path.exists(COLLAPSED_PATH):
        cube = np.load(COLLAPSED_PATH).astype(np.float64)
        log(f"loaded pre-collapsed cube {cube.shape} from {COLLAPSED_PATH}")
        return cube

    if not os.path.isfile(DATA_PATH):
        raise SystemExit(
            f"Missing hyperspectral data: {DATA_PATH}. Place the raw cube in "
            "data/hyperspectral or set SPOOL_HYPERSPECTRAL_RAW."
        )
    log(f"loading {DATA_PATH} ...")
    a = np.load(DATA_PATH, mmap_mode="r")
    log(f"  raw shape {a.shape}, dtype {a.dtype}")

    # crop first (the full 1024x1024x133x26 is 7 GB)
    H, W = a.shape[Y_AX], a.shape[X_AX]
    if CROP_SIZE is not None:
        S = min(CROP_SIZE, H, W)
        if CROP_CORNER == "center":
            y0, x0 = (H - S)//2, (W - S)//2
        elif CROP_CORNER == "bottom-left":
            y0, x0 = H - S, 0
        else:
            y0, x0 = 0, 0
        a = a[y0:y0+S, x0:x0+S]
        log(f"  cropped {S}x{S} from {CROP_CORNER} (y={y0}, x={x0})")

    # collapse the decay axis: time is used only to provide photons, not as an
    # estimation channel, so the spectral model sees exactly one physical channel
    cube = np.asarray(a, dtype=np.float64).sum(axis=T_AX)   # -> (Y, X, lambda)
    log(f"  collapsed over the decay axis -> {cube.shape} (Y, X, lambda)")
    return cube


# ---------------------------------------------------------------- panels
def save_panel(img, mask, cmap, vmin, vmax, name, scalebar_um=None, bar_color="white"):
    """Save one clean image panel. If scalebar_um is given, a bar of that physical
    length is drawn in the bottom-right corner."""
    disp = np.ma.array(img, mask=~mask)
    h, w = img.shape; dpi = 200
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)

    if scalebar_um is not None:
        bar_px = scalebar_um * 1000.0 / PIXEL_NM
        if bar_px > 0.9 * w:
            log(f"  [warn] {scalebar_um} um bar is {bar_px:.0f} px, wider than the "
                f"{w} px image; not drawn on {name}")
        else:
            # drawn in pixel (data) coordinates; imshow already sets the limits,
            # and calling set_xlim/set_ylim here would add a stray line at the edge
            pad_x, pad_y = 0.06*w, 0.08*h
            x1 = w - pad_x
            ax.plot([x1 - bar_px, x1], [h - pad_y]*2, color=bar_color,
                    lw=max(2.0, 0.012*h), solid_capstyle="butt",
                    clip_on=False, zorder=10)

    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=dpi); plt.close(fig)
    log(f"  saved {name}.png")


def save_colorbar(cmap, vmin, vmax, label, name):
    fig, ax = plt.subplots(figsize=(1.3, 4.5))
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=ax); cb.set_label(label, fontsize=13)
    cb.ax.tick_params(labelsize=11)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    log(f"Backend: {BACKEND}")
    os.makedirs(OUT_DIR, exist_ok=True)

    cube = load_cube()
    H, W, L = cube.shape
    lam, mu, S = build_spectral_dictionary(L)
    Ssum = S.sum(axis=1)
    K = S.shape[0]

    sigma_px = estimate_psf_sigma_px()
    psf = make_psf(sigma_px)
    o = make_otf(psf, (H, W))
    ones = (torch.ones((K, H, W), dtype=torch.float64, device=DEVICE)
            if USE_TORCH else np.ones((K, H, W)))
    C = fft_op(ones, o) * to_dev(Ssum).reshape(K, 1, 1)

    intensity = cube.sum(axis=2)
    corner = cube[:30, :30, :]
    bg_full = corner.sum() / (corner.shape[0]*corner.shape[1]) / L

    bright = intensity[intensity > np.percentile(intensity, 80)].mean()
    FG = intensity > FG_THRESHOLD_FRAC * bright
    full_mean = intensity[FG].mean()
    log(f"  foreground {FG.sum()}/{FG.size}, full-photon mean {full_mean:.1f} ph/px")

    # each method's own full-photon reference, at the same iteration count
    log("computing full-photon references ...")
    t0 = time.time()
    REF_MLE   = mean_wavelength(pixelwise_mle(cube, S, Ssum, bg_full), mu)
    REF_JOINT = mean_wavelength(joint_recon(cube, S, Ssum, o, C, bg_full), mu)
    log(f"  done in {time.time()-t0:.0f}s")

    if WAVELENGTH_RANGE_NM is not None:
        lo, hi = WAVELENGTH_RANGE_NM
        # warn if a lot of the data falls outside the fixed window, since that
        # saturates the map and hides real structure
        out = float(((REF_JOINT[FG] < lo) | (REF_JOINT[FG] > hi)).mean())
        log(f"  wavelength colour scale (fixed): {lo:.0f}-{hi:.0f} nm "
            f"({out:.1%} of foreground pixels saturate)")
        if out > 0.10:
            log(f"  [warn] more than 10% of pixels lie outside the fixed range; "
                f"the data span {np.percentile(REF_JOINT[FG],1):.0f}-"
                f"{np.percentile(REF_JOINT[FG],99):.0f} nm (1st-99th pct)")
    else:
        lo = float(np.percentile(REF_JOINT[FG], COLOR_PCT[0]))
        hi = float(np.percentile(REF_JOINT[FG], COLOR_PCT[1]))
        log(f"  wavelength colour scale ({COLOR_PCT[0]}-{COLOR_PCT[1]} pct): "
            f"{lo:.0f}-{hi:.0f} nm")
    cmap = plt.cm.rainbow.copy(); cmap.set_bad("black")
    cmap_err = plt.cm.inferno.copy(); cmap_err.set_bad("black")
    err_lo, err_hi = ERROR_RANGE_NM

    rows = []
    for target in SWEEP_PHOTONS:
        p = min(target/full_mean, 1.0)
        mean_ph = (intensity*p)[FG].mean()
        e_m, e_j, n_m, n_j = [], [], [], []
        disp_m = disp_j = mask_m = mask_j = None
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(7000 + trial)
            Yt = rng.binomial(cube.astype(np.int64), p).astype(np.float64)
            bg = bg_full * p
            Am = pixelwise_mle(Yt, S, Ssum, bg)
            Aj = joint_recon(Yt, S, Ssum, o, C, bg)
            wm = mean_wavelength(Am, mu); wj = mean_wavelength(Aj, mu)
            Mm = FG & (Am.sum(0) > ABUND_FRAC*Am.sum(0).max())
            Mj = FG & (Aj.sum(0) > ABUND_FRAC*Aj.sum(0).max())
            e_m.append(wm[Mm] - REF_MLE[Mm])
            e_j.append(wj[Mj] - REF_JOINT[Mj])
            n_m.append(int(Mm.sum())); n_j.append(int(Mj.sum()))
            if trial == 0:
                disp_m, disp_j, mask_m, mask_j = wm, wj, Mm, Mj

        # RMSE, reported alongside its bias and standard-deviation parts: the two
        # answer different questions and conflating them hides which one is moving
        em = np.concatenate(e_m); ej = np.concatenate(e_j)
        bm, sm = float(em.mean()), float(em.std())
        bj, sj = float(ej.mean()), float(ej.std())
        rm = float(np.sqrt(bm**2 + sm**2))
        rj = float(np.sqrt(bj**2 + sj**2))
        rows.append((mean_ph, rm, rj, bm, bj, sm, sj,
                     int(np.mean(n_m)), int(np.mean(n_j))))
        log(f"  {target:>2} ph/px ({mean_ph:.1f}): "
            f"MLE rmse {rm:.2f} nm (bias {bm:+.2f}, sd {sm:.2f}) [{int(np.mean(n_m))}px] | "
            f"Joint rmse {rj:.2f} nm (bias {bj:+.2f}, sd {sj:.2f}) [{int(np.mean(n_j))}px] | "
            f"{rm/rj:.1f}x")

        if target in DISPLAY_PHOTONS:
            save_panel(disp_m, mask_m, cmap, lo, hi, f"MLE_wavelength_{target}phpx")
            save_panel(disp_j, mask_j, cmap, lo, hi, f"Proposed_wavelength_{target}phpx")
            # |error| against each method's OWN full-photon reference
            save_panel(np.abs(disp_m - REF_MLE), mask_m, cmap_err, err_lo, err_hi,
                       f"MLE_abserror_{target}phpx")
            save_panel(np.abs(disp_j - REF_JOINT), mask_j, cmap_err, err_lo, err_hi,
                       f"Proposed_abserror_{target}phpx")

    save_panel(REF_MLE, FG, cmap, lo, hi, f"reference_MLE_{full_mean:.0f}phpx",
               scalebar_um=SCALEBAR_UM)
    save_panel(REF_JOINT, FG, cmap, lo, hi, f"reference_Proposed_{full_mean:.0f}phpx",
               scalebar_um=SCALEBAR_UM)
    save_colorbar(cmap, lo, hi, "Mean emission wavelength (nm)", "colorbar_wavelength")
    save_colorbar(cmap_err, err_lo, err_hi, "|Error| vs own reference (nm)", "colorbar_error")

    ph = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ph, [r[1] for r in rows], "o-", color="#888888", lw=2, ms=8, label="Pixel-wise MLE")
    ax.plot(ph, [r[2] for r in rows], "D-", color="#d62728", lw=2, ms=8, label="Proposed")
    ax.set_xscale("log")
    ax.set_xlabel("Mean photons per pixel", fontsize=14)
    ax.set_ylabel("Wavelength RMSE (nm)", fontsize=14)
    ax.tick_params(labelsize=12); ax.legend(fontsize=12); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "photon_sweep.png"), dpi=180)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("Hyperspectral: same estimator, spectral dictionary in place of the decay one.\n")
        f.write("Each method vs its OWN full-photon reconstruction, masked by its own abundance\n")
        f.write(f"(> {ABUND_FRAC:.0%} of that method's maximum). Pixel counts differ by construction.\n\n")
        f.write(f"{'ph/px':>7}{'MLE_rmse':>10}{'MLE_bias':>10}{'MLE_sd':>9}{'MLE_px':>9}"
                f"{'Prop_rmse':>11}{'Prop_bias':>11}{'Prop_sd':>9}{'Prop_px':>9}{'improve':>9}\n")
        for mp, rm, rj, bm, bj, sm, sj, nm_, nj_ in rows:
            f.write(f"{mp:>7.1f}{rm:>10.2f}{bm:>+10.2f}{sm:>9.2f}{nm_:>9}"
                    f"{rj:>11.2f}{bj:>+11.2f}{sj:>9.2f}{nj_:>9}{rm/rj:>9.2f}\n")
    log(f"\nDone. Panels in {OUT_DIR}")


if __name__ == "__main__":
    main()

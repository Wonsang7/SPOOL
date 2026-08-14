"""
FOV3 (cells) photon-efficiency panels at 1 and 5 photons/pixel.

Each method is compared against ITS OWN full-photon reconstruction:
  - MLE error      = |thinned MLE - full-photon MLE|
  - Proposed error = |thinned Proposed - full-photon Proposed|
so that the intrinsic PSF deconvolution of the proposed method is not
penalized by a detector-space (MLE) reference.

Outputs, per photon level (1 and 5 ph/px), each saved as its own PNG:
  reference lifetime (MLE), reference lifetime (Proposed), reference intensity,
  MLE lifetime, Proposed lifetime, MLE |error|, Proposed |error|,
  plus shared colorbars (lifetime, error, intensity).
"""

import os

# Windows/Anaconda: torch and NumPy(MKL) can each load their own Intel OpenMP
# runtime, which aborts with "OMP: Error #15". Must be set before the imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
import numpy as np
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors, matplotlib.cm

# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get(
    "SPOOL_FLIM_DATA_DIR", os.path.join(HERE, "data", "dual_labeled_flim")
)
OUT_DIR = os.environ.get(
    "SPOOL_FLIM_RESULTS_DIR", os.path.join(HERE, "results", "dual_labeled_flim")
)
SWEEP_PHOTONS   = [1, 3, 5, 7, 9, 11, 13, 15]   # sweep curve (same levels as Fig. 3)
DISPLAY_PHOTONS = [1, 5]                        # photon levels shown as image panels
N_TRIALS = 6
N_ITER = 50   # RL semi-convergence: see bead_microtubule script for the rationale
PIXEL_NM = 60.4                       # confocal pixel size for this acquisition
SCALEBAR_UM = 5.0                     # drawn on the reference panels (83 px here)
ETA = 0.9
DT_NS = 0.09696969697

# Unified PSF rule of the manuscript: FWHM = 0.51*lambda/NA with lambda = 550 nm,
# NA = 1.40 -> 200 nm, converted to detector pixels at this dataset's pitch.
# The value stored in fov3_calib.npy (2.00 px) predates this unification and is
# loaded below but NOT used.
PSF_FWHM_NM = 200.0
PSF_SIG_UNIFIED = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM   # -> 1.406 px

LIFE_LO, LIFE_HI = 2.4, 3.6           # lifetime colormap scale (ns). The cells sample
                                      # sits in a HIGHER lifetime band than the bead /
                                      # microtubule samples, so the 0.5-2.5 ns scale
                                      # used there would saturate almost everything.
ERR_LO, ERR_HI = 0.0, 2.0             # |error| colormap scale (ns)
ABUND_FRAC = 0.05                     # each method is scored/displayed only where its
                                      # OWN recovered abundance exceeds this fraction
                                      # of its own maximum
# ===============================================================

required_files = ["fov3_cube.npz", "fov3_calib.npy", "fov3_FULLIMG_mle.npz"]
missing_files = [name for name in required_files if not os.path.isfile(os.path.join(DATA, name))]
if missing_files:
    raise SystemExit(
        "Missing dual-labeled FLIM data files: " + ", ".join(missing_files)
        + ". Place them in data/dual_labeled_flim or set SPOOL_FLIM_DATA_DIR."
    )
os.makedirs(OUT_DIR, exist_ok=True)
d = np.load(f"{DATA}/fov3_cube.npz")
intensity = d["intensity"].astype(float); cube = d["cube"].astype(float)
IRF_T0, IRF_SIG, BG, PSF_SIG_CALIB, BGPB_full = np.load(f"{DATA}/fov3_calib.npy")
PSF_SIG = PSF_SIG_UNIFIED
print(f"PSF sigma: calib {PSF_SIG_CALIB:.3f} px (IGNORED) -> unified {PSF_SIG:.3f} px "
      f"({PSF_FWHM_NM:.0f} nm FWHM at {PIXEL_NM} nm pitch)")
FG = np.load(f"{DATA}/fov3_FULLIMG_mle.npz")["FG"]
# NOTE: the full-photon references are recomputed below at N_ITER and with the
# unified PSF rather than loaded from disk. The stored fov3_FULLIMG_*.npz
# references were produced with 150 iterations and the old calib sigma; reusing
# them would compare each method against a differently-converged, differently-
# modelled version of itself.
REF_MLE = REF_JOINT = None   # computed after the operators are defined

N_T = cube.shape[2]; t = np.arange(N_T) * DT_NS; EPS = 1e-9
TAU_GRID = np.round(np.arange(1.0, 5.01, 0.4), 2); K = len(TAU_GRID)

def mkb(tau):
    lam = 1/tau
    dd = np.clip(0.5*lam*np.exp(0.5*lam**2*IRF_SIG**2 - lam*(t-IRF_T0)), 0, None)
    return dd/dd.sum()
D = np.stack([mkb(x) for x in TAU_GRID])

def mkpsf(s, rf=4):
    r = int(np.ceil(rf*s)); a = np.arange(-r, r+1); xx, yy = np.meshgrid(a, a)
    h = np.exp(-(xx**2+yy**2)/(2*s**2)); return h/h.sum()
PSF = mkpsf(PSF_SIG)

# ---------------------------------------------------------------- backend
# PyTorch + CUDA if available, else NumPy on CPU with identical results.
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
print(f"Backend: {BACKEND}")


def to_dev(a):
    if USE_TORCH:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=DEVICE)
    return np.asarray(a, dtype=np.float64)


def to_np(a):
    if USE_TORCH and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# A plain FFT product is CIRCULAR convolution, which wraps the opposite image edge
# around and does NOT match fftconvolve(mode="same") at the borders. Zero-pad to the
# full linear-convolution size, multiply, then crop back -- exactly what scipy does.
# The adjoint uses the OTF of the FLIPPED PSF, not conj(OTF), which would add a shift.
def _make_otf(psf, shape):
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


def _einsum_khw_kt(A, Dx):
    return torch.einsum("khw,kt->hwt", A, Dx) if USE_TORCH else np.einsum("khw,kt->hwt", A, Dx)


def _einsum_hwt_kt(R, Dx):
    return torch.einsum("hwt,kt->khw", R, Dx) if USE_TORCH else np.einsum("hwt,kt->khw", R, Dx)


_D_d = to_dev(D)
_Dsum_d = to_dev(D.sum(axis=1)).reshape(K, 1, 1)
_H, _W = cube.shape[:2]
_OTF = _make_otf(PSF, (_H, _W))
_ones = (torch.ones((K, _H, _W), dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.ones((K, _H, _W)))
_C = _fft_op(_ones, _OTF) * _Dsum_d


def pmle(Y, bgpb, ni=N_ITER):
    """Pixel-wise Poisson MLE (h -> Dirac delta). Returns a NumPy array."""
    Yd = to_dev(Y)
    h, w, _ = Y.shape
    init = max(float(np.asarray(Y).sum())/(h*w*K), 1e-4)
    A = (torch.full((K, h, w), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, h, w), init))
    for _ in range(ni):
        lam = _einsum_khw_kt(A, _D_d) + bgpb
        R = Yd/(lam + EPS)
        A = A * (_einsum_hwt_kt(R, _D_d) / (_Dsum_d + EPS))
    return to_np(A)


def joint(Y, bgpb, ni=N_ITER, eta=ETA):
    """Joint spatial(PSF)-temporal(decay basis) Poisson reconstruction."""
    Yd = to_dev(Y)
    h, w, _ = Y.shape
    init = max(float(np.asarray(Y).sum())/(h*w*K), 1e-4)
    A = (torch.full((K, h, w), init, dtype=torch.float64, device=DEVICE)
         if USE_TORCH else np.full((K, h, w), init))
    for _ in range(ni):
        blur = _fft_op(A, _OTF)
        lam = _einsum_khw_kt(blur, _D_d) + bgpb
        R = Yd/(lam + EPS)
        U = _fft_op(_einsum_hwt_kt(R, _D_d), _OTF, adjoint=True)
        ratio = U/(_C + EPS)
        ratio = torch.clamp(ratio, min=0.0) if USE_TORCH else np.clip(ratio, 0, None)
        A = A * ratio**eta
        A = torch.clamp(A, min=1e-8) if USE_TORCH else np.maximum(A, 1e-8)
    return to_np(A)


def mean_tau(A):
    tot = A.sum(0)
    return np.divide((TAU_GRID[:, None, None]*A).sum(0), tot, out=np.zeros_like(tot), where=tot > EPS)

# ---- full-photon references, computed at the SAME iteration count as everything else ----
print(f"computing full-photon references at {N_ITER} iterations ...")
_t0 = time.time()
REF_MLE = mean_tau(pmle(cube, BGPB_full))
REF_JOINT = mean_tau(joint(cube, BGPB_full))
print(f"  done in {time.time()-_t0:.0f}s")


def save_panel(img, mask, cmap, vmin, vmax, name, scalebar_um=None, bar_color="white"):
    """Save a clean image panel. If scalebar_um is given, a bar of that physical
    length is drawn in the bottom-right corner. `bar_color` matters on the hot
    intensity map, which saturates to white -- use a contrasting color there."""
    disp = np.ma.array(img, mask=~mask)
    h, w = img.shape; dpi = 200
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)

    if scalebar_um is not None:
        bar_px = scalebar_um * 1000.0 / PIXEL_NM
        if bar_px > 0.9 * w:
            print(f"  [warn] {scalebar_um} um bar is {bar_px:.0f} px, wider than the "
                  f"{w} px image; not drawn on {name}")
        else:
            # Drawn in pixel (data) coordinates. imshow already sets the limits to
            # (0..w, h..0); calling set_xlim/set_ylim here would add a stray line
            # along the image edge.
            pad_x, pad_y = 0.06 * w, 0.08 * h
            x1 = w - pad_x
            ax.plot([x1 - bar_px, x1], [h - pad_y] * 2, color=bar_color,
                    lw=max(2.0, 0.012 * h), solid_capstyle="butt",
                    clip_on=False, zorder=10)

    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=dpi); plt.close(fig)
    print(f"  saved {name}.png")

def save_colorbar(cmap, vmin, vmax, label, name):
    fig, ax = plt.subplots(figsize=(1.3, 4.5))
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=ax); cb.set_label(label, fontsize=13); cb.ax.tick_params(labelsize=11)
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

cmap_life = plt.cm.rainbow.copy(); cmap_life.set_bad("black")
cmap_err = plt.cm.inferno.copy(); cmap_err.set_bad("black")
cmap_int = plt.cm.hot.copy(); cmap_int.set_bad("black")
full_mean_ph = intensity[FG].mean()

# references (saved once)
save_panel(REF_MLE, FG, cmap_life, LIFE_LO, LIFE_HI,
           f"reference_MLE_lifetime_{full_mean_ph:.0f}phpx", scalebar_um=SCALEBAR_UM)
save_panel(REF_JOINT, FG, cmap_life, LIFE_LO, LIFE_HI,
           f"reference_SPOOL_lifetime_{full_mean_ph:.0f}phpx", scalebar_um=SCALEBAR_UM)
save_panel(intensity, FG, cmap_int, 0, np.percentile(intensity[FG], 99),
           f"reference_intensity_{full_mean_ph:.0f}phpx",
           scalebar_um=SCALEBAR_UM, bar_color="cyan")

summary = []
for target in SWEEP_PHOTONS:
    p = min(target / full_mean_ph, 1.0)
    mean_ph = (intensity * p)[FG].mean()
    mle_errs, joint_errs = [], []
    n_mle_px, n_joint_px = [], []
    disp_mle = disp_joint = None
    mask_mle = mask_joint = None
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(4000 + trial)
        Yt = rng.binomial(cube.astype(np.int64), p).astype(np.float64)
        bgpb = BGPB_full * p
        A_m = pmle(Yt, bgpb)
        A_j = joint(Yt, bgpb)
        tm = mean_tau(A_m)
        tj = mean_tau(A_j)

        # Each method is masked by ITS OWN recovered abundance: a pixel where a
        # method reconstructed no signal carries no lifetime information, so
        # scoring it would penalize an estimate that was never made. The two
        # methods are therefore evaluated on different pixel counts (reported).
        ab_m = A_m.sum(0); ab_j = A_j.sum(0)
        Mm = FG & (ab_m > ABUND_FRAC * ab_m.max())
        Mj = FG & (ab_j > ABUND_FRAC * ab_j.max())

        mle_errs.append(tm[Mm] - REF_MLE[Mm])       # vs own full-photon MLE
        joint_errs.append(tj[Mj] - REF_JOINT[Mj])   # vs own full-photon SPOOL
        n_mle_px.append(int(Mm.sum())); n_joint_px.append(int(Mj.sum()))
        if trial == 0:
            disp_mle, disp_joint = tm, tj
            mask_mle, mask_joint = Mm, Mj
    me = np.concatenate(mle_errs); je = np.concatenate(joint_errs)
    # RMSE, reported alongside its bias and standard-deviation parts. The bound
    # in Section 2.5 constrains variance, and RMSE mixes the two, so quoting only
    # the combination would hide which of them is actually moving.
    bm, sm = float(me.mean()), float(me.std())
    bj, sj = float(je.mean()), float(je.std())
    rm = float(np.sqrt(bm**2 + sm**2))
    rj = float(np.sqrt(bj**2 + sj**2))
    summary.append((mean_ph, rm, rj, bm, bj, sm, sj,
                    int(np.mean(n_mle_px)), int(np.mean(n_joint_px))))
    print(f"level {target} ph/px (actual {mean_ph:.1f}): "
          f"MLE rmse {rm:.3f} (bias {bm:+.3f}, sd {sm:.3f}) [{int(np.mean(n_mle_px))}px] | "
          f"SPOOL rmse {rj:.3f} (bias {bj:+.3f}, sd {sj:.3f}) [{int(np.mean(n_joint_px))}px] | "
          f"{rm/rj:.1f}x")
    if target in DISPLAY_PHOTONS:
        tag = f"{target}phpx"
        save_panel(disp_mle, mask_mle, cmap_life, LIFE_LO, LIFE_HI, f"MLE_lifetime_{tag}")
        save_panel(disp_joint, mask_joint, cmap_life, LIFE_LO, LIFE_HI, f"SPOOL_lifetime_{tag}")
        save_panel(np.abs(disp_mle - REF_MLE), mask_mle, cmap_err, ERR_LO, ERR_HI, f"MLE_abserror_{tag}")
        save_panel(np.abs(disp_joint - REF_JOINT), mask_joint, cmap_err, ERR_LO, ERR_HI,
                   f"SPOOL_abserror_{tag}")

# ---- photon-sweep curve (same style and photon levels as the bead/microtubule figure) ----
ph   = [s[0] for s in summary]
rm_l = [s[1] for s in summary]
rj_l = [s[2] for s in summary]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(ph, rm_l, "o-", color="#888888", lw=2, ms=8, label="MLE")
ax.plot(ph, rj_l, "D-", color="#d62728", lw=2, ms=8, label="SPOOL")
ax.set_xscale("log")
ax.set_xlabel("Mean photons per pixel", fontsize=14)
ax.set_ylabel("Lifetime RMSE (ns)", fontsize=14)
ax.tick_params(labelsize=12); ax.legend(fontsize=12); ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/photon_sweep.png", dpi=180); plt.close(fig)
print("  saved photon_sweep.png")

save_colorbar(cmap_life, LIFE_LO, LIFE_HI, "Lifetime (ns)", "colorbar_lifetime")
save_colorbar(cmap_err, ERR_LO, ERR_HI, "|Error| vs own reference (ns)", "colorbar_error")
save_colorbar(cmap_int, 0, np.percentile(intensity[FG], 99), "Photon count", "colorbar_intensity")

with open(f"{OUT_DIR}/summary.txt", "w") as f:
    f.write("FOV3 cells: each method vs its OWN full-photon reference,\n")
    f.write(f"masked by its OWN recovered abundance (> {ABUND_FRAC:.0%} of that method's max).\n")
    f.write("Pixel counts differ between methods by construction and are reported.\n")
    f.write(f"N_ITER = {N_ITER}, N_TRIALS = {N_TRIALS}, "
            f"PSF sigma = {PSF_SIG:.2f} px (unified 200 nm FWHM rule; "
            f"calib value {PSF_SIG_CALIB:.2f} px ignored)\n\n")
    f.write(f"{'ph/px':>7}{'MLE_rmse':>10}{'MLE_bias':>10}{'MLE_sd':>9}{'MLE_px':>9}"
            f"{'Prop_rmse':>11}{'Prop_bias':>11}{'Prop_sd':>9}{'Prop_px':>9}{'improve':>9}\n")
    for mp, rm, rj, bm, bj, sm, sj, nm, nj in summary:
        f.write(f"{mp:>7.1f}{rm:>10.3f}{bm:>+10.3f}{sm:>9.3f}{nm:>9}"
                f"{rj:>11.3f}{bj:>+11.3f}{sj:>9.3f}{nj:>9}{rm/rj:>9.2f}\n")
print(f"\nAll panels saved to {OUT_DIR}")

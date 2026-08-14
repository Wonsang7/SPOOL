"""
GPU-accelerated photon-efficiency panels (PyTorch backend, full field of view).

Uses PyTorch + CUDA if available, otherwise falls back to NumPy on CPU with
identical numerical results. The reconstruction math is unchanged from the
CPU reference implementation; the speedup comes from (a) the GPU and (b)
batching the per-component spatial convolutions into a single FFT call
instead of looping over components in Python.

Each method is compared against ITS OWN full-photon reconstruction:
    MLE error      = thinned MLE      - full-photon MLE
    Proposed error = thinned Proposed - full-photon Proposed
so the proposed method's intrinsic PSF deconvolution is not penalized by a
detector-space (MLE) reference.

Each method is also MASKED BY ITS OWN recovered abundance (above ABUND_FRAC of
that method's own maximum), for both the statistics and the displayed images: a
pixel where a method reconstructed no signal carries no lifetime information, so
scoring it would penalize an estimate that was never made. Because the proposed
method concentrates signal via PSF deconvolution, its mask is tighter than the
MLE's, and the two methods are therefore scored on different pixel counts. Those
counts are written to the summary file, and this choice MUST be stated in the
Methods section of any write-up.

REQUIREMENTS: torch with CUDA (verified working via
    python -c "import torch; print(torch.cuda.is_available())"   -> True
No CuPy needed.

USAGE:
    python bead_microtubule_panels_torch.py
"""

import os

# ---------------------------------------------------------------------------
# Windows/Anaconda: PyTorch and NumPy(MKL) can each pull in their own copy of the
# Intel OpenMP runtime (libiomp5md.dll), which aborts with "OMP: Error #15".
# This must be set BEFORE numpy/torch are imported. It is the standard (if
# officially unsupported) workaround; we additionally pin the thread count,
# which avoids the oversubscription that makes the duplicate runtime dangerous.
# Results are validated against the single-threaded CPU reference implementation.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Pinning threads avoids the oversubscription that makes a duplicated OpenMP
# runtime risky. It costs nothing when the work runs on the GPU. If you end up
# on the CPU fallback and want multi-threading back, comment the next line out.
os.environ.setdefault("OMP_NUM_THREADS", "1")
# ---------------------------------------------------------------------------

import time
import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import tifffile

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
        BACKEND = "NumPy / CPU (torch present but CUDA unavailable)"
except ImportError:
    BACKEND = "NumPy / CPU (torch not installed)"

DTYPE = np.float64


def to_dev(a):
    """NumPy array -> backend tensor."""
    if USE_TORCH:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=DEVICE)
    return np.asarray(a, dtype=DTYPE)


def to_np(a):
    """Backend tensor -> NumPy array."""
    if USE_TORCH and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SPOOL_BEAD_MICROTUBULE_DATA_DIR",
                          os.path.join(HERE, "data", "bead_microtubule"))
SAMPLES = {
    "bead":        dict(tif="Confocal_yellow_green_beads_488_px30nm.tif",
                        pixel_nm=30.0, crop=100),
    "microtubule": dict(tif="Confocal_microtubule_Oregon_green_px39nm.tif",
                        pixel_nm=39.0, crop=200),
}
OUT_DIR = os.environ.get("SPOOL_BEAD_MICROTUBULE_RESULTS_DIR",
                         os.path.join(HERE, "results", "bead_microtubule"))

CROP_CORNER     = "bottom-left"                 # which corner the crop is taken from
SCALEBAR_UM     = 1.0                           # scale bar length, drawn on the
                                                # highest-photon image of each sample.
                                                # (5 um would not fit: the bead crop is
                                                # only 100 px x 30 nm = 3.0 um wide.)
SWEEP_PHOTONS   = [1, 3, 5, 7, 9, 11, 13, 15]
DISPLAY_PHOTONS = [1, 5, 15]
DT_NS       = 0.0977
PSF_FWHM_NM = 200.0
TAU_MIN, TAU_MAX, TAU_STEP = 1.0, 5.0, 0.4
N_TRIALS    = 6
N_ITER      = 50
ETA         = 0.9
FG_THRESHOLD_FRAC = 0.15
ABUND_FRAC        = 0.05    # each method is scored/displayed only where its OWN
                            # recovered abundance exceeds this fraction of its own
                            # maximum (see note in the sweep loop)
LIFE_LO, LIFE_HI  = 0, 3                    # lifetime colormap scale (ns)
EPS = 1e-9
# ===============================================================


def log(*a):
    print(*a, flush=True)


def load_cube(tif, crop_size):
    path = os.path.join(DATA_DIR, tif)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing experimental data: {path}. Place the TIFF files in "
            "data/bead_microtubule or set SPOOL_BEAD_MICROTUBULE_DATA_DIR."
        )
    arr = tifffile.imread(path).astype(np.float64)  # (T,Y,X)
    cube = np.moveaxis(arr, 0, -1)                                          # (Y,X,T)
    if crop_size is not None:
        H, W = cube.shape[:2]
        S = min(crop_size, H, W)
        # image coordinates: y increases downwards, so "bottom" = the last rows
        if CROP_CORNER == "bottom-left":
            y0, x0 = H - S, 0
        elif CROP_CORNER == "bottom-right":
            y0, x0 = H - S, W - S
        elif CROP_CORNER == "top-left":
            y0, x0 = 0, 0
        elif CROP_CORNER == "top-right":
            y0, x0 = 0, W - S
        elif CROP_CORNER == "center":
            y0, x0 = (H - S) // 2, (W - S) // 2
        else:
            raise ValueError(f"Unknown CROP_CORNER: {CROP_CORNER}")
        cube = cube[y0:y0+S, x0:x0+S, :]
        log(f"  cropped {S}x{S} from {CROP_CORNER} (y={y0}:{y0+S}, x={x0}:{x0+S})")
    intensity = cube.sum(axis=2)
    log(f"  cube {cube.shape}, intensity mean {intensity.mean():.1f} ph/px")
    return cube, intensity


def fit_irf(cube):
    prof = cube.sum(axis=(0, 1)); N_T = len(prof)
    t = np.arange(N_T) * DT_NS; peak = int(np.argmax(prof))

    def model(t, amp, t0, sig, tau, bg):
        lam = 1.0 / tau
        arg = (t0 + lam*sig**2 - t) / (np.sqrt(2)*sig)
        ex = np.clip(0.5*lam**2*sig**2 - lam*(t - t0), -700, 700)
        return amp*0.5*lam*np.exp(ex)*erfc(arg) + bg

    p0 = [prof.max()*2, t[peak]-0.2, 0.1, 3.0, prof[:max(1, peak-5)].mean()]
    try:
        popt, _ = curve_fit(model, t, prof, p0=p0, maxfev=20000)
        irf_t0, irf_sig = popt[1], abs(popt[2])
        if irf_sig < 0.5*DT_NS or not np.isfinite(irf_sig):
            irf_sig = 1.5*DT_NS
    except Exception:
        irf_t0, irf_sig = t[peak], 2*DT_NS
    log(f"  IRF t0={irf_t0:.3f} ns, sigma={irf_sig:.3f} ns")
    return irf_t0, irf_sig


def build_operators(N_T, irf_t0, irf_sig, pixel_nm):
    t = np.arange(N_T) * DT_NS
    tau_grid = np.round(np.arange(TAU_MIN, TAU_MAX + 1e-9, TAU_STEP), 3)

    def basis(tau):
        lam = 1.0/tau
        arg = (irf_t0 + lam*irf_sig**2 - t) / (np.sqrt(2)*irf_sig)
        d = np.clip(0.5*lam*np.exp(0.5*lam**2*irf_sig**2 - lam*(t - irf_t0))*erfc(arg), 0, None)
        s = d.sum()
        return d/s if s > 0 else d

    D = np.stack([basis(tt) for tt in tau_grid])
    sigma_px = (PSF_FWHM_NM / 2.3548200450309493) / pixel_nm
    r = int(np.ceil(4*sigma_px)); a = np.arange(-r, r+1)
    weights = 0.5 * (erf((a + 0.5) / (np.sqrt(2) * sigma_px))
                     - erf((a - 0.5) / (np.sqrt(2) * sigma_px)))
    psf = np.outer(weights, weights); psf /= psf.sum()
    predicted_gain = 1.0 / np.sqrt(psf[r, r])
    log(f"  {len(tau_grid)} bases, pixel-integrated PSF sigma={sigma_px:.2f} px, "
        f"predicted gain={predicted_gain:.2f}x")
    return tau_grid, D, psf


# ---------------------------------------------------------------- FFT ops
# A plain FFT product gives CIRCULAR convolution, which wraps the opposite edge
# of the image around and does NOT match scipy's fftconvolve(mode="same") at the
# borders. We therefore zero-pad to the full linear-convolution size, multiply in
# Fourier space, and crop back -- exactly what scipy does internally. The adjoint
# uses the OTF of the FLIPPED PSF (not conj(OTF), which would add a circular shift).
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
    """Batched zero-padded convolution of every component map in A (K,H,W)."""
    fh, fw = o["fshape"]; H, W = o["out"]; y0, x0 = o["off"]
    G = o["otf_adj"] if adjoint else o["otf"]
    if USE_TORCH:
        F = torch.fft.rfft2(A, s=(fh, fw))
        full = torch.fft.irfft2(F * G.unsqueeze(0), s=(fh, fw))
    else:
        F = np.fft.rfft2(A, s=(fh, fw), axes=(1, 2))
        full = np.fft.irfft2(F * G[None, :, :], s=(fh, fw), axes=(1, 2))
    return full[:, y0:y0+H, x0:x0+W]


def conv_batch(A, o):
    return _fft_op(A, o, adjoint=False)


def corr_batch(A, o):
    return _fft_op(A, o, adjoint=True)


# ---------------------------------------------------------------- recon
def _einsum_khw_kt(A, D):
    """(K,H,W) x (K,T) -> (H,W,T)"""
    if USE_TORCH:
        return torch.einsum("khw,kt->hwt", A, D)
    return np.einsum("khw,kt->hwt", A, D)


def _einsum_hwt_kt(R, D):
    """(H,W,T) x (K,T) -> (K,H,W)"""
    if USE_TORCH:
        return torch.einsum("hwt,kt->khw", R, D)
    return np.einsum("hwt,kt->khw", R, D)


def _full(shape, val):
    if USE_TORCH:
        return torch.full(shape, float(val), dtype=torch.float64, device=DEVICE)
    return np.full(shape, val, dtype=DTYPE)


def _clip_min(A, lo):
    return torch.clamp(A, min=lo) if USE_TORCH else np.maximum(A, lo)


def _clip0(A):
    return torch.clamp(A, min=0.0) if USE_TORCH else np.clip(A, 0, None)


def pixelwise_mle(Y, D, bgpb, n_iter=N_ITER):
    """Pixel-wise Poisson MLE (h -> Dirac delta; no spatial coupling)."""
    K = D.shape[0]; H, W, T = Y.shape
    total = float(to_np(Y).sum())
    A = _full((K, H, W), max(total/(H*W*K), 1e-4))
    Dsum = D.sum(axis=1) if not USE_TORCH else D.sum(dim=1)
    for _ in range(n_iter):
        lam = _einsum_khw_kt(A, D) + bgpb
        ratio = Y / (lam + EPS)
        back = _einsum_hwt_kt(ratio, D)                 # (K,H,W), all components at once
        A = A * (back / (Dsum.reshape(K, 1, 1) + EPS))
    return A


def joint_recon(Y, D, o, bgpb, C, n_iter=N_ITER, eta=ETA):
    """Joint spatial(PSF)-temporal(decay basis) Poisson reconstruction."""
    K = D.shape[0]; H, W, T = Y.shape
    total = float(to_np(Y).sum())
    A = _full((K, H, W), max(total/(H*W*K), 1e-4))
    for _ in range(n_iter):
        blur = conv_batch(A, o)                          # one batched FFT
        lam = _einsum_khw_kt(blur, D) + bgpb
        ratio = Y / (lam + EPS)
        back = _einsum_hwt_kt(ratio, D)                  # (K,H,W)
        U = corr_batch(back, o)                          # adjoint PSF, batched
        A = A * _clip0(U / (C + EPS)) ** eta
        A = _clip_min(A, 1e-8)
    return A


def mean_tau(A, tau_grid_d):
    tot = A.sum(0)
    num = (tau_grid_d.reshape(-1, 1, 1) * A).sum(0)
    if USE_TORCH:
        return torch.where(tot > EPS, num / (tot + EPS), torch.zeros_like(tot))
    return np.where(tot > EPS, num / (tot + EPS), 0.0)


# ---------------------------------------------------------------- plotting
def save_panel(img, FG, cmap, vmin, vmax, path, pixel_nm=None, scalebar_um=None):
    """Save one clean image panel. If pixel_nm and scalebar_um are both given, a
    scale bar of that physical length is drawn in the bottom-right corner."""
    disp = np.ma.array(img, mask=~FG)
    h, w = img.shape; dpi = 200
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)

    if pixel_nm is not None and scalebar_um is not None:
        bar_px = scalebar_um * 1000.0 / pixel_nm      # physical length -> pixels
        if bar_px > 0.9 * w:
            log(f"  [warn] {scalebar_um} um scale bar is {bar_px:.0f} px, wider than "
                f"the {w} px image; not drawn on {os.path.basename(path)}")
        else:
            # Draw in pixel (data) coordinates, inset from the bottom-right corner.
            # imshow's default limits are already (0..w, h..0), so we must not
            # touch set_xlim/set_ylim -- doing so adds a stray line at the edge.
            pad_x = 0.06 * w
            pad_y = 0.08 * h
            y = h - pad_y
            x1 = w - pad_x
            x0 = x1 - bar_px
            ax.plot([x0, x1], [y, y], color="white",
                    lw=max(2.0, 0.025 * h), solid_capstyle="butt",
                    clip_on=False, zorder=10)

    fig.savefig(path, dpi=dpi); plt.close(fig)


def save_colorbar(cmap, vmin, vmax, label, path):
    fig, ax = plt.subplots(figsize=(1.3, 4.5))
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=ax)
    cb.set_label(label, fontsize=13); cb.ax.tick_params(labelsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def process_sample(name, cfg):
    log(f"\n=== {name} ===")
    sample_out = os.path.join(OUT_DIR, name)
    os.makedirs(sample_out, exist_ok=True)

    cube_np, intensity_np = load_cube(cfg["tif"], cfg["crop"])
    N_T = cube_np.shape[2]
    irf_t0, irf_sig = fit_irf(cube_np)
    tau_grid, D_np, psf_np = build_operators(N_T, irf_t0, irf_sig, cfg["pixel_nm"])

    corner = cube_np[:30, :30, :]
    bgpb_full = corner.sum() / (corner.shape[0]*corner.shape[1]) / N_T

    FG_np = intensity_np >= FG_THRESHOLD_FRAC * float(intensity_np.max())
    full_mean_ph = intensity_np[FG_np].mean()
    log(f"  foreground {FG_np.sum()}/{FG_np.size}, full-photon mean {full_mean_ph:.1f} ph/px")

    H, W = cube_np.shape[:2]
    K = D_np.shape[0]
    cube_d = to_dev(cube_np); D_d = to_dev(D_np); tau_d = to_dev(tau_grid)
    o = make_otf(psf_np, (H, W))

    ones = _full((K, H, W), 1.0)
    Dsum_d = to_dev(D_np.sum(axis=1)).reshape(K, 1, 1)
    C = conv_batch(ones, o) * Dsum_d
    h0 = float(psf_np[psf_np.shape[0] // 2, psf_np.shape[1] // 2])
    predicted_gain = float(1.0 / np.sqrt(h0))

    t0 = time.time()
    REF_MLE = to_np(mean_tau(pixelwise_mle(cube_d, D_d, bgpb_full), tau_d))
    REF_JOINT = to_np(mean_tau(joint_recon(cube_d, D_d, o, bgpb_full, C), tau_d))
    log(f"  full-photon references done in {time.time()-t0:.1f}s")

    cmap_life = plt.cm.rainbow.copy(); cmap_life.set_bad("black")
    rows = []
    for target in SWEEP_PHOTONS:
        p = min(target / full_mean_ph, 1.0)
        mean_ph = (intensity_np * p)[FG_np].mean()
        mle_errs, joint_errs = [], []
        n_mle_px, n_joint_px = [], []
        disp_mle = disp_joint = None
        mask_mle = mask_joint = None
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(1000 + trial)
            Yt = to_dev(rng.binomial(cube_np.astype(np.int64), p).astype(np.float64))
            bgpb = bgpb_full * p
            A_mle_d = pixelwise_mle(Yt, D_d, bgpb)
            A_joint_d = joint_recon(Yt, D_d, o, bgpb, C)
            tm = to_np(mean_tau(A_mle_d, tau_d))
            tj = to_np(mean_tau(A_joint_d, tau_d))

            # Each method is masked by ITS OWN recovered signal: a pixel is only
            # scored where that method actually reconstructed abundance above
            # ABUND_FRAC of its own maximum. Pixels where a method recovered no
            # signal carry no lifetime information, so including them would score
            # an estimate that was never made. Note this means the two methods are
            # evaluated on different pixel sets (counts are reported below), a
            # direct consequence of the proposed method concentrating signal via
            # PSF deconvolution -- this must be stated in the Methods.
            Mm = FG_np & (to_np(A_mle_d.sum(0)) > ABUND_FRAC * float(to_np(A_mle_d.sum(0)).max()))
            Mj = FG_np & (to_np(A_joint_d.sum(0)) > ABUND_FRAC * float(to_np(A_joint_d.sum(0)).max()))

            mle_errs.append(tm[Mm] - REF_MLE[Mm])
            joint_errs.append(tj[Mj] - REF_JOINT[Mj])
            n_mle_px.append(int(Mm.sum())); n_joint_px.append(int(Mj.sum()))
            if trial == 0:
                disp_mle, disp_joint = tm, tj
                mask_mle, mask_joint = Mm, Mj
        me = np.concatenate(mle_errs); je = np.concatenate(joint_errs)
        row = dict(target=target, mean_ph=mean_ph,
                   mle_std=float(np.std(me)), joint_std=float(np.std(je)),
                   n_mle_px=int(np.mean(n_mle_px)), n_joint_px=int(np.mean(n_joint_px)))
        rows.append(row)
        if target in DISPLAY_PHOTONS:
            # the scale bar goes on the highest-photon image of this sample only
            sb = SCALEBAR_UM if target == max(DISPLAY_PHOTONS) else None
            px = cfg["pixel_nm"] if sb is not None else None
            save_panel(disp_mle, mask_mle, cmap_life, LIFE_LO, LIFE_HI,
                       os.path.join(sample_out, f"{name}_MLE_ph{target}.png"),
                       pixel_nm=px, scalebar_um=sb)
            save_panel(disp_joint, mask_joint, cmap_life, LIFE_LO, LIFE_HI,
                       os.path.join(sample_out, f"{name}_Proposed_ph{target}.png"),
                       pixel_nm=px, scalebar_um=sb)
        log(f"  ph~{target} ({mean_ph:.1f}): MLE sd {row['mle_std']:.3f} [{row['n_mle_px']}px] "
            f"Proposed sd {row['joint_std']:.3f} [{row['n_joint_px']}px] "
            f"({row['mle_std']/row['joint_std']:.1f}x)")

    ph = [r["mean_ph"] for r in rows]
    mle_exponent = float(np.polyfit(np.log(ph), np.log([r["mle_std"] for r in rows]), 1)[0])
    spool_exponent = float(np.polyfit(np.log(ph), np.log([r["joint_std"] for r in rows]), 1)[0])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ph, [r["mle_std"] for r in rows], "o-", color="#888888", lw=2, ms=8, label="MLE")
    ax.plot(ph, [r["joint_std"] for r in rows], "D-", color="#d62728", lw=2, ms=8, label="Proposed")
    ax.set_xscale("log")
    ax.set_xlabel("Mean photons per pixel", fontsize=14)
    ax.set_ylabel("Lifetime precision, s.d. (ns)", fontsize=14)
    ax.tick_params(labelsize=12); ax.legend(fontsize=12); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(sample_out, f"{name}_photon_sweep.png"), dpi=180)
    plt.close(fig)

    save_colorbar(cmap_life, LIFE_LO, LIFE_HI, "Lifetime (ns)",
                  os.path.join(sample_out, f"{name}_colorbar_lifetime.png"))

    with open(os.path.join(sample_out, f"{name}_summary.txt"), "w") as f:
        f.write(f"{name}: MLE vs Proposed\n")
        f.write("Each method vs its own full-photon reference, and masked by its own\n")
        f.write(f"recovered abundance (> {ABUND_FRAC:.0%} of that method's own maximum).\n")
        f.write("Pixel counts differ between methods by construction and are reported.\n\n")
        f.write(f"Pixel pitch: {cfg['pixel_nm']:.1f} nm; PSF FWHM: {PSF_FWHM_NM:.1f} nm; "
                f"pixels/FWHM: {PSF_FWHM_NM/cfg['pixel_nm']:.2f}.\n")
        f.write(f"Predicted gain 1/sqrt(h0): {predicted_gain:.3f}x.\n")
        f.write(f"Log-log photon-scaling exponents: MLE {mle_exponent:.3f}, "
                f"SPOOL {spool_exponent:.3f}.\n\n")
        f.write(f"{'target':>7}{'mean_ph':>9}{'MLE_sd':>9}{'MLE_px':>9}"
                f"{'Prop_sd':>9}{'Prop_px':>9}{'improve':>9}\n")
        for r in rows:
            f.write(f"{r['target']:>7}{r['mean_ph']:>9.1f}{r['mle_std']:>9.3f}{r['n_mle_px']:>9}"
                    f"{r['joint_std']:>9.3f}{r['n_joint_px']:>9}"
                    f"{r['mle_std']/r['joint_std']:>9.2f}\n")
    log(f"  panels -> {sample_out}")


def main():
    log(f"Backend: {BACKEND}")
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    for name, cfg in SAMPLES.items():
        process_sample(name, cfg)
    log(f"\nAll done in {time.time()-t0:.0f}s. Panels in {OUT_DIR}")


if __name__ == "__main__":
    main()

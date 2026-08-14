"""
oracle_pooled_control.py
========================
Adds the two controls the manuscript currently lacks, and which a reviewer will
ask for immediately.

THE OBJECTION
    "If the whole gain comes from pooling photons that share a parameter, why not
     just sum the photons in the emitter's footprint and fit one decay? Or bin the
     image 3x3? Neither requires a deconvolution."

This is correct, and it must be answered with data rather than rhetoric. Two
controls are added to the single-emitter sweep:

  ORACLE ROI-POOLED MLE.  Knowing the emitter's position exactly, sum the decay
      histograms over the whole PSF footprint into a single histogram and fit
      (A, tau) by continuous Poisson maximum likelihood -- no dictionary, no
      deconvolution, no spatial regularisation. This estimator has Fisher
      information N*J and should therefore SIT ON the pooled Cramer-Rao bound.
      It is an oracle: it needs the emitter's location and its support, and it
      returns ONE number, not a map. It cannot be run on an extended object.

  3x3 SPATIAL BINNING + pixel-wise MLE.  What a practitioner actually does. Gains
      at most sqrt(9) = 3x and costs a factor of 3 in spatial resolution.

EXPECTED OUTCOME, and the claim it licenses
    oracle-pooled ~ pooled CRB  <  proposed  <<  3x3-binned  <  pixel-wise
    i.e. the bound IS attainable in this setting, which retroactively validates
    using it as a benchmark; the proposed estimator comes close to the oracle
    WITHOUT being told where the emitter is and WITHOUT giving up spatial
    resolution; and naive binning recovers only a fraction of the gain.

    The honest sentence this supports is:
        "The proposed estimator approaches the performance of an oracle that is
         told the emitter's location, while requiring no such knowledge and
         retaining a spatially resolved reconstruction."
    which is both weaker and far more defensible than "the pixel-wise estimator
    is efficiently solving the wrong problem".

NOTE ON THE PARAMETERISATION MISMATCH
    The pooled bound is derived for a 2-parameter model (N, tau) with a
    continuous lifetime and a known emitter location. The proposed estimator
    solves a different problem: K nonnegative abundances per pixel over the whole
    field, on a finite lifetime grid, with early stopping. The bound is therefore
    an ORACLE bound for a parametric isolated emitter, not "the exact CRB of the
    proposed estimator", and the manuscript must say so. This control makes the
    distinction concrete: the oracle estimator is the one the bound actually
    describes, and it is included so the reader can see the gap the proposed
    estimator has to cross.

    python oracle_pooled_control.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu":
    print("[warn] no CUDA - running on CPU, this will be slow")
torch.backends.cuda.matmul.allow_tf32 = True

# ---- must match single_bead_crb_mc.py exactly -------------------------------
HERE        = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.environ.get("SPOOL_SINGLE_EMITTER_RESULTS_DIR",
                             os.path.join(HERE, "results", "single_emitter"))
FIELD       = 48
PIXEL_NM    = 50
PSF_FWHM_NM = 200.0
N_T         = 256
DT_NS       = 0.0977
TAU_TRUE    = 2.0
TAU_MIN, TAU_MAX, TAU_STEP = 1.0, 5.0, 0.4
IRF_SIG_NS  = 0.0637
IRF_T0_NS   = 0.50
BG_PER_BIN  = 1e-4
FG_FRAC     = 0.15
MEAN_PH_SWEEP = [1, 3, 5, 7, 9, 11, 13, 15]
N_MC        = 400
MC_CHUNK    = 64
JOINT_N_ITER = 50
MLE_N_ITER   = 1000
ETA, EPS, SEED = 0.9, 1e-8, 20260712
DT = torch.float32

ROI_SIGMA   = 3.0     # oracle ROI radius, in units of sigma (captures 98.9% of a Gaussian)
BIN         = 3       # spatial binning factor for the practitioner control


def decay(tau, t, t0, sig):
    tau = np.atleast_1d(np.asarray(tau, float))[:, None]
    tt = np.asarray(t, float)[None, :]
    lam = 1.0 / tau
    arg = (t0 + lam * sig ** 2 - tt) / (np.sqrt(2) * sig)
    ex = np.clip(0.5 * lam ** 2 * sig ** 2 - lam * (tt - t0), -700, 700)
    d = np.clip(0.5 * lam * np.exp(ex) * erfc(arg), 0.0, None)
    s = d.sum(1, keepdims=True)
    return d / np.where(s > 0, s, 1.0)


def gaussian_psf(s):
    """Pixel-integrated, unit-sum 2D Gaussian PSF (Eq. S1)."""
    r = int(np.ceil(4 * s)); a = np.arange(-r, r + 1)
    w = 0.5 * (erf((a + 0.5) / (np.sqrt(2) * s))
               - erf((a - 0.5) / (np.sqrt(2) * s)))
    p = np.outer(w, w)
    return p / p.sum()


def crb_pooled(tau0, n_tot, h_flat, t, t0, sig, bg, h=1e-4):
    d = decay(tau0, t, t0, sig)[0]
    dp = (decay(tau0 + h, t, t0, sig)[0] - decay(tau0 - h, t, t0, sig)[0]) / (2 * h)
    hx = h_flat[:, None]
    mu = n_tot * hx * d[None, :] + bg + 1e-15
    gN, gT = hx * d[None, :], n_tot * hx * dp[None, :]
    I_NN = (gN * gN / mu).sum(); I_NT = (gN * gT / mu).sum(); I_TT = (gT * gT / mu).sum()
    return I_NN / (I_NN * I_TT - I_NT ** 2)


def crb_pixel(tau0, n_c, t, t0, sig, bg, h=1e-4):
    d = decay(tau0, t, t0, sig)[0]
    dp = (decay(tau0 + h, t, t0, sig)[0] - decay(tau0 - h, t, t0, sig)[0]) / (2 * h)
    mu = n_c * d + bg + 1e-15
    gA, gT = d, n_c * dp
    I_AA = (gA * gA / mu).sum(); I_AT = (gA * gT / mu).sum(); I_TT = (gT * gT / mu).sum()
    return I_AA / (I_AA * I_TT - I_AT ** 2)


# ---------------------------------------------------------------- estimators
def pooled_mle_tau(hist, t, bg_tot):
    """CONTINUOUS Poisson ML fit of a single (A, tau) to one summed histogram.

    This is the estimator the pooled bound actually describes: two parameters, a
    continuous lifetime, no dictionary. A is profiled out analytically (for a
    normalised basis and a known background, the ML amplitude is just the total
    signal count), leaving a 1-D search over tau.
    """
    Ntot = max(hist.sum() - bg_tot * len(hist), 1e-6)

    def nll(tau):
        if not (TAU_MIN * 0.5 < tau < TAU_MAX * 1.5):
            return 1e12
        mu = Ntot * decay(tau, t, IRF_T0_NS, IRF_SIG_NS)[0] + bg_tot
        mu = np.maximum(mu, 1e-12)
        return float((mu - hist * np.log(mu)).sum())

    r = minimize_scalar(nll, bounds=(0.3, 8.0), method="bounded",
                        options=dict(xatol=1e-4))
    return float(r.x)


def make_otf(psf, shape):
    H, W = shape; ph, pw = psf.shape
    fh, fw = H + ph - 1, W + pw - 1
    pad = np.zeros((fh, fw)); pad[:ph, :pw] = psf
    padf = np.zeros((fh, fw)); padf[:ph, :pw] = psf[::-1, ::-1]
    return dict(otf=torch.fft.rfft2(torch.as_tensor(pad, dtype=DT, device=DEV), s=(fh, fw)),
                adj=torch.fft.rfft2(torch.as_tensor(padf, dtype=DT, device=DEV), s=(fh, fw)),
                fs=(fh, fw), out=(H, W), off=((ph - 1) // 2, (pw - 1) // 2))


def _op(A, o, adj):
    fh, fw = o["fs"]; H, W = o["out"]; y0, x0 = o["off"]
    G = o["adj"] if adj else o["otf"]
    full = torch.fft.irfft2(torch.fft.rfft2(A, s=(fh, fw)) * G, s=(fh, fw))
    return full[..., y0:y0 + H, x0:x0 + W]


def lam_of(A, D):
    B, K, H, W = A.shape
    return (A.permute(0, 2, 3, 1).reshape(-1, K) @ D).reshape(B, H, W, -1)


def back_of(R, Dt):
    B, H, W, T = R.shape
    return (R.reshape(-1, T) @ Dt).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


def mle_batch(Y, D, Dt, Dsum, bg, n_iter):
    B, H, W, T = Y.shape; K = D.shape[0]
    A = torch.full((B, K, H, W), 1e-2, dtype=DT, device=DEV)
    for _ in range(n_iter):
        A = A * (back_of(Y / (lam_of(A, D) + bg + EPS), Dt) / (Dsum + EPS))
    return A


def joint_batch(Y, D, Dt, o, bg, C, n_iter):
    B, H, W, T = Y.shape; K = D.shape[0]
    A = torch.full((B, K, H, W), 1e-2, dtype=DT, device=DEV)
    for _ in range(n_iter):
        lam = lam_of(_op(A, o, False), D) + bg
        U = _op(back_of(Y / (lam + EPS), Dt), o, True)
        A = torch.clamp(A * torch.clamp(U / (C + EPS), min=0.0) ** ETA, min=1e-10)
    return A


def tau_at(A, tau_g, cy, cx):
    a = A[:, :, cy, cx]; tot = a.sum(1)
    return torch.where(tot > 1e-8, (a * tau_g).sum(1) / (tot + EPS),
                       torch.zeros_like(tot)).double().cpu().numpy()


# ---------------------------------------------------------------- main
def main():
    gen = torch.Generator(device=DEV); gen.manual_seed(SEED)
    t = np.arange(N_T) * DT_NS
    sig_px = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM
    psf = gaussian_psf(sig_px)

    H = W = FIELD; cy = cx = FIELD // 2
    h_map = np.zeros((H, W)); r = psf.shape[0] // 2
    h_map[cy - r:cy + r + 1, cx - r:cx + r + 1] = psf
    h0 = float(h_map[cy, cx])
    FG = h_map >= FG_FRAC * h0
    hbar = float(h_map[FG].mean())

    # oracle ROI: a disc of ROI_SIGMA * sigma around the KNOWN emitter position
    yy, xx = np.mgrid[0:H, 0:W]
    ROI = np.hypot(yy - cy, xx - cx) <= ROI_SIGMA * sig_px
    roi_frac = float(h_map[ROI].sum())
    print(f"sigma {sig_px:.2f} px | 1/h(0) = {1/h0:.1f} -> predicted gain {np.sqrt(1/h0):.2f}x")
    print(f"oracle ROI: {int(ROI.sum())} px, capturing {100*roi_frac:.1f}% of the emitter's photons\n")

    tau_grid = np.round(np.arange(TAU_MIN, TAU_MAX + 1e-9, TAU_STEP), 4)
    K = len(tau_grid)
    D_np = decay(tau_grid, t, IRF_T0_NS, IRF_SIG_NS)
    d_true = decay(TAU_TRUE, t, IRF_T0_NS, IRF_SIG_NS)[0]

    D = torch.as_tensor(D_np, dtype=DT, device=DEV).contiguous()
    Dt = D.t().contiguous(); Dsum = D.sum(1).reshape(1, K, 1, 1)
    tau_g = torch.as_tensor(tau_grid, dtype=DT, device=DEV).reshape(1, K)
    o = make_otf(psf, (H, W))
    C = _op(torch.ones((1, K, H, W), dtype=DT, device=DEV), o, False) * Dsum
    shape = torch.as_tensor(h_map[:, :, None] * d_true[None, None, :], dtype=DT, device=DEV)
    h_flat = h_map.ravel()

    print(f"{'mean':>5} {'CRBpx':>7} {'MLE':>7} {'bin3':>7} {'Prop':>7} {'oracle':>7} "
          f"{'CRBpool':>8} | {'orac/CRB':>9} {'prop/CRB':>9} {'prop/orac':>10}")
    rows = []
    for mp in MEAN_PH_SWEEP:
        N_tot = mp / hbar
        mu = N_tot * shape + BG_PER_BIN
        tm, tj, to_, tb = [], [], [], []
        done = 0
        while done < N_MC:
            b = min(MC_CHUNK, N_MC - done)
            Y = torch.poisson(mu.expand(b, H, W, N_T).contiguous(), generator=gen)

            tm.append(tau_at(mle_batch(Y, D, Dt, Dsum, BG_PER_BIN, MLE_N_ITER),
                             tau_g, cy, cx))
            tj.append(tau_at(joint_batch(Y, D, Dt, o, BG_PER_BIN, C, JOINT_N_ITER),
                             tau_g, cy, cx))

            Yn = Y.double().cpu().numpy()

            # --- oracle ROI-pooled MLE: one summed histogram, continuous tau ---
            for i in range(b):
                hist = Yn[i][ROI].sum(axis=0)
                to_.append(pooled_mle_tau(hist, t, BG_PER_BIN * int(ROI.sum())))

            # --- 3x3 binned pixel-wise MLE, read at the emitter's bin ---
            hb = (H // BIN) * BIN
            Yb = Yn[:, :hb, :hb, :].reshape(b, hb // BIN, BIN, hb // BIN, BIN, N_T).sum((2, 4))
            Yb_t = torch.as_tensor(Yb, dtype=DT, device=DEV)
            Ab = mle_batch(Yb_t, D, Dt, Dsum, BG_PER_BIN * BIN * BIN, MLE_N_ITER)
            tb.append(tau_at(Ab, tau_g, cy // BIN, cx // BIN))
            del Ab, Yb_t, Y
            done += b
        torch.cuda.empty_cache()

        f = lambda a: np.concatenate(a) if isinstance(a, list) and isinstance(a[0], np.ndarray) else np.asarray(a)
        tm, tj, tb = f(tm), f(tj), f(tb)
        to_ = np.asarray(to_)
        ok = tm > 1e-6
        sd_m = float(tm[ok].std(ddof=1))
        sd_j = float(tj[tj > 1e-6].std(ddof=1))
        sd_b = float(tb[tb > 1e-6].std(ddof=1))
        sd_o = float(to_.std(ddof=1))

        c_px = float(np.sqrt(crb_pixel(TAU_TRUE, N_tot * h0, t, IRF_T0_NS, IRF_SIG_NS, BG_PER_BIN)))
        c_pl = float(np.sqrt(crb_pooled(TAU_TRUE, N_tot, h_flat, t, IRF_T0_NS, IRF_SIG_NS, BG_PER_BIN)))

        rows.append(dict(mean=mp, crb_px=c_px, mle=sd_m, bin3=sd_b, prop=sd_j,
                         orac=sd_o, crb_pool=c_pl))
        print(f"{mp:>5} {c_px:>7.3f} {sd_m:>7.3f} {sd_b:>7.3f} {sd_j:>7.3f} {sd_o:>7.3f} "
              f"{c_pl:>8.3f} | {sd_o/c_pl:>9.2f} {sd_j/c_pl:>9.2f} {sd_j/sd_o:>10.2f}")

    print("\nREAD THIS AS:")
    print("  orac/CRB ~ 1  -> the pooled bound IS attainable here, which is what licenses")
    print("                   using it as a benchmark at all. It is an ORACLE: it needs the")
    print("                   emitter's position and returns one number, not a map.")
    print("  prop/orac      -> how much the proposed estimator gives up by NOT being told")
    print("                   where the emitter is, and by returning a full spatial map.")
    print("  bin3           -> what naive spatial binning buys, at 3x the pixel size.")

    x = [r["mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot(x, [r["crb_px"] for r in rows], "--", c="#888888", lw=1.5, label="CRB, single pixel")
    ax.plot(x, [r["crb_pool"] for r in rows], "--", c="#d62728", lw=1.5, label="CRB, PSF-footprint pooling")
    ax.plot(x, [r["mle"] for r in rows], "o-", c="#888888", lw=2, ms=6, label="Pixel-wise MLE")
    ax.plot(x, [r["bin3"] for r in rows], "^-", c="#7f7f7f", lw=1.6, ms=6, ls=":",
            label=f"{BIN}x{BIN}-binned MLE")
    ax.plot(x, [r["orac"] for r in rows], "s-", c="#1f77b4", lw=1.8, ms=6,
            label="Oracle ROI-pooled MLE")
    ax.plot(x, [r["prop"] for r in rows], "D-", c="#d62728", lw=2, ms=6, label="Proposed")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mean photons per pixel", fontsize=13)
    ax.set_ylabel("Lifetime s.d. across realizations (ns)", fontsize=13)
    ax.legend(fontsize=8.5, frameon=False); ax.grid(alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "oracle_pooled_control.png"), dpi=200)
    print(f"\n-> {OUT_DIR}/oracle_pooled_control.png")


if __name__ == "__main__":
    main()

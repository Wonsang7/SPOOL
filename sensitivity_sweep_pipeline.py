"""
Robustness / sensitivity sweep, built directly on single_bead_crb_mc.py.

Everything that defines the measurement is taken unchanged from that script:
the Gaussian PSF, the IRF-convolved decay basis, the dictionary, the pixel-wise
MLE, the joint spatio-temporal estimator, the ROI-pooled oracle, the 3x3-binned
control, the analytic CRBs, and the read-at-the-emitter-pixel convention. The
only additions are three mismatch axes and the loop that sweeps them.

    A. SBR sweep        SBR in {5, 10, 20, 50},   PSF err = 0,  bkg err = 0
    B. PSF FWHM error   SBR = 10, err in {-20,-10,0,+10,+20}%,  bkg err = 0
    C. background error SBR = 10, PSF err = 0,    err in {-20, 0, +20}%
    D. worst-case       SBR = 5,  PSF err = +20%, bkg err = -20%

What "mismatch" means here
-------------------------
* PSF error: the estimators are handed a PSF of FWHM (1+e) times the truth.
  The data are always generated with the true PSF. The joint estimator's OTF
  and its normaliser C are built from the assumed PSF, and the oracle's ROI
  radius is set from the assumed sigma, because a practitioner sizing that ROI
  would only have the assumed PSF to go on.
* Background error: the estimators are handed (1+e) times the true background
  rate. The data are always generated with the true rate.
* The CRBs are computed from the TRUE PSF and TRUE background throughout. They
  are properties of the data, not of the assumed model, so they do not move
  when the assumed PSF does. That is the point: any change in the measured
  s.d. under mismatch is lost estimator efficiency, not lost information.

Photon budget and SBR
---------------------
As in the original script, N_tot = mean_photons / hbar, where hbar is the mean
of the normalised PSF over the FG_FRAC mask, so "mean photons per pixel" is a
foreground quantity. SBR is therefore

    SBR = mean_photons_per_pixel / (bg_per_bin * N_T)

and the sweep sets bg_per_bin = mean_photons / (SBR * N_T). The nominal
single-emitter analysis instead uses a fixed background of 1e-4 counts per
pixel per bin, corresponding to an SBR of approximately 39 at one photon per
foreground pixel. The sweep therefore brackets that nominal condition.

Statistics
----------
N_MC realisations per condition are split into N_BLOCK blocks. The s.d. is
computed within each block, giving N_BLOCK independent estimates whose mean
and standard error are what the figure plots. Bias and RMSE are taken against
TAU_TRUE.

Output
------
results_sensitivity.json, in the schema sensitivity_figure.py expects (one row
per condition x photon level x method x block, with the block index in the
"seed" field).

Usage
-----
    python sensitivity_sweep_pipeline.py                     # 1 ph/px
    python sensitivity_sweep_pipeline.py --photons 1,5,15
    python sensitivity_sweep_pipeline.py --n-mc 800 --no-oracle
"""

import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import minimize_scalar
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu":
    print("[warn] no CUDA - running on CPU, this will be slow")
torch.backends.cuda.matmul.allow_tf32 = True

# ---- unchanged from single_bead_crb_mc.py -----------------------------------
FIELD       = 48
PIXEL_NM    = 50
PSF_FWHM_NM = 200.0
N_T         = 256
DT_NS       = 0.0977
TAU_TRUE    = 2.0
TAU_MIN, TAU_MAX, TAU_STEP = 1.0, 5.0, 0.4
IRF_SIG_NS  = 0.0637
IRF_T0_NS   = 0.50
FG_FRAC     = 0.15
JOINT_N_ITER = 50
MLE_N_ITER   = 1000
ETA, EPS, SEED = 0.9, 1e-8, 20260712
DT = torch.float32
ROI_SIGMA   = 3.0
BIN         = 3

# ---- sweep-specific ---------------------------------------------------------
N_MC_DEFAULT = 480
N_BLOCK      = 8
MC_CHUNK     = 64


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


def pooled_mle_tau(hist, t, bg_tot):
    """Continuous 2-parameter Poisson ML fit to one summed histogram."""
    Ntot = max(hist.sum() - bg_tot * len(hist), 1e-6)

    def nll(tau):
        if not (TAU_MIN * 0.5 < tau < TAU_MAX * 1.5):
            return 1e12
        mu = Ntot * decay(tau, t, IRF_T0_NS, IRF_SIG_NS)[0] + bg_tot
        return float((np.maximum(mu, 1e-12) - hist * np.log(np.maximum(mu, 1e-12))).sum())

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


# ---------------------------------------------------------------- conditions
def condition_grid():
    conds = []

    def add(tag, sbr, psf_err, bkg_err):
        key = (round(sbr, 6), round(psf_err, 6), round(bkg_err, 6))
        for c in conds:
            if (round(c["sbr"], 6), round(c["psf_err"], 6), round(c["bkg_err"], 6)) == key:
                c["tags"].append(tag)
                return
        conds.append(dict(tags=[tag], sbr=sbr, psf_err=psf_err, bkg_err=bkg_err))

    for s in (5.0, 10.0, 20.0, 50.0):
        add("sbr", s, 0.0, 0.0)
    for e in (-0.20, -0.10, 0.0, 0.10, 0.20):
        add("psf", 10.0, e, 0.0)
    for e in (-0.20, 0.0, 0.20):
        add("bkg", 10.0, 0.0, e)
    add("corner", 5.0, 0.20, -0.20)
    for i, c in enumerate(conds):
        c["cid"] = i
    return conds


# ---------------------------------------------------------------- one cell
def run_cell(cond, mean_ph, shared, n_mc, chunk, do_oracle, do_bin):
    t, D, Dt, Dsum, tau_g, sig_true, h_map, FG, hbar, h0, h_flat, shape = (
        shared[k] for k in ("t", "D", "Dt", "Dsum", "tau_g", "sig_true", "h_map",
                            "FG", "hbar", "h0", "h_flat", "shape"))
    H = W = FIELD; cy = cx = FIELD // 2

    sig_a = sig_true * (1.0 + cond["psf_err"])
    psf_a = gaussian_psf(sig_a)
    o = make_otf(psf_a, (H, W))
    C = _op(torch.ones((1, D.shape[0], H, W), dtype=DT, device=DEV), o, False) * Dsum

    yy, xx = np.mgrid[0:H, 0:W]
    ROI = np.hypot(yy - cy, xx - cx) <= ROI_SIGMA * sig_a

    bg_true = mean_ph / (cond["sbr"] * N_T)          # per pixel per bin
    bg_a = bg_true * (1.0 + cond["bkg_err"])
    N_tot = mean_ph / hbar
    mu = N_tot * shape + bg_true

    # Common random numbers. The seed depends only on the photon level, NOT on
    # the condition, so every condition that generates data from the same mu
    # sees the *identical* Poisson realisations. The PSF and background sweeps
    # hold the true model fixed and perturb only what the estimator is told, so
    # their mu is identical and the sweep becomes an exactly paired comparison:
    # the curve then measures the mismatch, not the Monte Carlo noise. The SBR
    # sweep genuinely changes mu, so it stays unpaired and needs more
    # realisations to reach the same precision.
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED + int(round(mean_ph * 10)))

    out = {k: [] for k in ("pixelwise", "proposed")}
    if do_oracle:
        out["pooled"] = []
    if do_bin:
        out["bin3"] = []

    done = 0
    while done < n_mc:
        b = min(chunk, n_mc - done)
        Y = torch.poisson(mu.expand(b, H, W, N_T).contiguous(), generator=gen)
        out["pixelwise"].append(
            tau_at(mle_batch(Y, D, Dt, Dsum, bg_a, MLE_N_ITER), tau_g, cy, cx))
        out["proposed"].append(
            tau_at(joint_batch(Y, D, Dt, o, bg_a, C, JOINT_N_ITER), tau_g, cy, cx))
        if do_oracle or do_bin:
            Yn = Y.double().cpu().numpy()
        if do_oracle:
            n_roi = int(ROI.sum())
            out["pooled"].append(np.array(
                [pooled_mle_tau(Yn[i][ROI].sum(axis=0), t, bg_a * n_roi) for i in range(b)]))
        if do_bin:
            hb = (H // BIN) * BIN
            Yb = Yn[:, :hb, :hb, :].reshape(b, hb // BIN, BIN, hb // BIN, BIN, N_T).sum((2, 4))
            Ab = mle_batch(torch.as_tensor(Yb, dtype=DT, device=DEV),
                           D, Dt, Dsum, bg_a * BIN * BIN, MLE_N_ITER)
            out["bin3"].append(tau_at(Ab, tau_g, cy // BIN, cx // BIN))
            del Ab
        del Y
        done += b
    if DEV.type == "cuda":
        torch.cuda.empty_cache()

    # CRBs: true PSF, true background -- invariant to the assumed model
    crb_px = float(np.sqrt(crb_pixel(TAU_TRUE, N_tot * h0, t, IRF_T0_NS, IRF_SIG_NS, bg_true)))
    crb_pl = float(np.sqrt(crb_pooled(TAU_TRUE, N_tot, h_flat, t, IRF_T0_NS, IRF_SIG_NS, bg_true)))

    rows = []
    per = n_mc // N_BLOCK
    for m, chunks in out.items():
        v = np.concatenate(chunks)
        for blk in range(N_BLOCK):
            s = v[blk * per:(blk + 1) * per]
            s = s[s > 1e-6]
            if s.size < 3:
                continue
            e = s - TAU_TRUE
            rows.append(dict(method=m, seed=blk,
                             sd_gt=float(s.std(ddof=1)),
                             bias_gt=float(e.mean()),
                             rmse_gt=float(np.sqrt((e ** 2).mean())),
                             n_gt=int(s.size)))
    for m, val in (("crb_pixel", crb_px), ("crb_pooled", crb_pl)):
        for blk in range(N_BLOCK):
            rows.append(dict(method=m, seed=blk, sd_gt=val, bias_gt=0.0,
                             rmse_gt=val, n_gt=n_mc))
    return rows, dict(bg_per_bin=bg_true, n_tot=N_tot, crb_pixel=crb_px,
                      crb_pooled=crb_pl, sigma_assumed=sig_a, roi_px=int(ROI.sum()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", default="1", help="comma-separated, e.g. 1,5,15")
    ap.add_argument("--n-mc", type=int, default=N_MC_DEFAULT)
    ap.add_argument("--chunk", type=int, default=MC_CHUNK)
    ap.add_argument("--out", default="results_sensitivity.json")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--no-bin", action="store_true")
    ap.add_argument("--mle-iters", type=int, default=None,
                    help="override MLE_N_ITER (testing only; the pipeline uses 1000)")
    a = ap.parse_args()
    if a.mle_iters:
        global MLE_N_ITER
        MLE_N_ITER = a.mle_iters
        print(f'[warn] MLE_N_ITER overridden to {MLE_N_ITER} (pipeline value is 1000)')
    photons = [float(x) for x in a.photons.split(",")]
    n_mc = (a.n_mc // N_BLOCK) * N_BLOCK
    if n_mc // N_BLOCK < 8:
        raise SystemExit(f"--n-mc must be at least {8 * N_BLOCK} so that each of the "
                         f"{N_BLOCK} blocks has enough realisations for an s.d.")

    t = np.arange(N_T) * DT_NS
    sig_true = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM
    psf = gaussian_psf(sig_true)
    H = W = FIELD; cy = cx = FIELD // 2
    h_map = np.zeros((H, W)); r = psf.shape[0] // 2
    h_map[cy - r:cy + r + 1, cx - r:cx + r + 1] = psf
    h0 = float(h_map[cy, cx])
    FG = h_map >= FG_FRAC * h0
    hbar = float(h_map[FG].mean())

    tau_grid = np.round(np.arange(TAU_MIN, TAU_MAX + 1e-9, TAU_STEP), 4)
    D_np = decay(tau_grid, t, IRF_T0_NS, IRF_SIG_NS)
    d_true = decay(TAU_TRUE, t, IRF_T0_NS, IRF_SIG_NS)[0]
    D = torch.as_tensor(D_np, dtype=DT, device=DEV).contiguous()
    shared = dict(
        t=t, D=D, Dt=D.t().contiguous(),
        Dsum=D.sum(1).reshape(1, len(tau_grid), 1, 1),
        tau_g=torch.as_tensor(tau_grid, dtype=DT, device=DEV).reshape(1, len(tau_grid)),
        sig_true=sig_true, h_map=h_map, FG=FG, hbar=hbar, h0=h0, h_flat=h_map.ravel(),
        shape=torch.as_tensor(h_map[:, :, None] * d_true[None, None, :], dtype=DT, device=DEV))

    print(f"sigma {sig_true:.4f} px (FWHM {2.35482 * sig_true:.2f} px) | "
          f"1/h(0) = {1 / h0:.2f} -> Eq. (10) predicted gain {np.sqrt(1 / h0):.2f}x")
    print(f"dictionary K = {len(tau_grid)}, tau_true = {TAU_TRUE} ns, "
          f"{n_mc} MC in {N_BLOCK} blocks\n")

    conds = condition_grid()
    rows, meta = [], {}
    t0 = time.time()
    job, n_jobs = 0, len(conds) * len(photons)
    for cond in conds:
        for mp in photons:
            job += 1
            rr, mm = run_cell(cond, mp, shared, n_mc, a.chunk,
                              not a.no_oracle, not a.no_bin)
            for r in rr:
                r.update(cid=cond["cid"], tags=cond["tags"], sbr=cond["sbr"],
                         psf_err=cond["psf_err"], bkg_err=cond["bkg_err"],
                         sigma_assumed=mm["sigma_assumed"], photons=mp)
            rows += rr
            meta[f"{cond['cid']}_{mp}"] = mm
            sd = {m: float(np.mean([r["sd_gt"] for r in rr if r["method"] == m]))
                  for m in set(x["method"] for x in rr)}
            sp, sj = sd.get("pixelwise", np.nan), sd.get("proposed", np.nan)
            g = sp / sj if np.isfinite(sp) and np.isfinite(sj) and sj > 0 else np.nan
            el = time.time() - t0
            print(f"[{job:3d}/{n_jobs}] SBR={cond['sbr']:>4.0f} psf={cond['psf_err']:+.0%} "
                  f"bkg={cond['bkg_err']:+.0%} n={mp:>4.1f} | sd_px={sp:.3f} "
                  f"sd_prop={sj:.3f} gain={g:.2f}x | {el:5.0f}s, "
                  f"~{el / job * (n_jobs - job):5.0f}s left", flush=True)

    payload = dict(
        config=dict(sigma_true_px=sig_true, fwhm_px=2.35482 * sig_true,
                    field=FIELD, pixel_nm=PIXEL_NM, psf_fwhm_nm=PSF_FWHM_NM,
                    n_t=N_T, dt_ns=DT_NS, tau_true=TAU_TRUE,
                    tau_grid=tau_grid.tolist(), irf_sig_ns=IRF_SIG_NS,
                    irf_t0_ns=IRF_T0_NS, fg_frac=FG_FRAC,
                    joint_n_iter=JOINT_N_ITER, mle_n_iter=MLE_N_ITER, eta=ETA,
                    roi_sigma=ROI_SIGMA, bin=BIN, n_mc=n_mc, n_block=N_BLOCK,
                    predicted_gain_eq10=float(np.sqrt(1 / h0))),
        conditions=conds, seeds=N_BLOCK, per_cell=meta,
        backend=DEV.type, rows=rows)
    with open(a.out, "w") as fh:
        json.dump(payload, fh)
    print(f"\nwrote {a.out}  ({len(rows)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

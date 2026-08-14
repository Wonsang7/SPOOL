"""
iter_optimization_sim.py
========================
Iteration-count optimization for the joint estimator, on the single-emitter
simulation of oracle_pooled_control.py, where the TRUE lifetime is known.

METRIC
    Per-realization lifetime at the emitter pixel, over N_MC realizations:
        bias(n_iter) = mean(tau_hat) - TAU_TRUE
        sd(n_iter)   = std(tau_hat, ddof=1)
        rmse(n_iter) = sqrt(bias^2 + sd^2)
    The selected iteration count is argmin RMSE (minimum-risk stopping,
    cf. predictive-risk stopping for Poisson EM, Massa & Benvenuto 2021,
    and the bias-vs-SD iteration curves standard in PET quantification).

DESIGN
    * ONE RL run to max(ITER_CHECKPOINTS); tau is snapshotted at each
      checkpoint. All iteration counts therefore see IDENTICAL photon
      realizations -- exact common-random-numbers pairing at zero extra cost.
    * A clean simulation flatters large iteration counts (no model error to
      amplify), so the scan is repeated under model-mismatch conditions in
      the spirit of SS8: the DATA are generated with the nominal PSF while
      the PSF supplied to the RECONSTRUCTION is perturbed. The recommended N is one
      whose RMSE is at (or within one SE of) the minimum under BOTH the
      nominal and the mismatch conditions.
    * Total photon budget is held fixed across conditions (normalized by the
      nominal footprint), so conditions differ only in model error.

Place NEXT TO oracle_pooled_control.py and run:
    python iter_optimization_sim.py
"""

import os
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the model, operators, and config of the CRB validation script.
# Imported defensively: local copies of oracle_pooled_control.py may differ
# slightly (e.g. no log()), so missing names are reported all at once.
import oracle_pooled_control as opc


def log(*a):
    print(*a, flush=True)


_NEEDED = ["decay", "gaussian_psf", "make_otf", "_op", "lam_of", "back_of",
           "tau_at", "DEV", "DT",
           "FIELD", "PIXEL_NM", "PSF_FWHM_NM", "N_T", "DT_NS",
           "TAU_TRUE", "TAU_MIN", "TAU_MAX", "TAU_STEP",
           "IRF_SIG_NS", "IRF_T0_NS", "BG_PER_BIN",
           "FG_FRAC", "N_MC", "MC_CHUNK", "ETA", "EPS", "SEED"]
_missing = [n for n in _NEEDED if not hasattr(opc, n)]
if _missing:
    raise ImportError(
        f"oracle_pooled_control.py is missing: {_missing}. "
        "Make sure the version alongside this script is the one with the "
        "pixel-integrated PSF / oracle controls (the Figure-1a version).")
for _n in _NEEDED:
    globals()[_n] = getattr(opc, _n)

# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("SPOOL_ITERATION_RESULTS_DIR",
                         os.path.join(HERE, "results", "iteration_selection"))

# selection figure: iterations 10-300 over the paper's full photon range
ITER_CHECKPOINTS = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]

# full sweep range of the paper (1-15 photons/px)
MEAN_PH_LEVELS = [1, 3, 5, 7, 9, 11, 13, 15]

# model-mismatch conditions: reconstruction PSF sigma scale factors.
# data generation always uses the NOMINAL PSF. 1.0 = no mismatch.
# The heatmap (panel a) is drawn for "nominal"; panel (b) overlays all.
PSF_SCALE_CONDITIONS = {
    "nominal":  1.00,
    "psf+10%":  1.10,
    "psf-10%":  0.90,
}
# ===============================================================


def joint_checkpoint_taus(Y, D, Dt, o, bg, C, checkpoints, tau_g, cy, cx):
    """One RL run to max(checkpoints); returns {n_iter: tau array (B,)}.

    Snapshotting inside a single run guarantees every iteration count is
    evaluated on the same realizations (exact CRN pairing).
    """
    B, H, W, T = Y.shape
    K = D.shape[0]
    cps = set(checkpoints)
    A = torch.full((B, K, H, W), 1e-2, dtype=DT, device=DEV)
    out = {}
    for it in range(1, max(checkpoints) + 1):
        lam = lam_of(_op(A, o, False), D) + bg
        U = _op(back_of(Y / (lam + EPS), Dt), o, True)
        A = torch.clamp(A * torch.clamp(U / (C + EPS), min=0.0) ** ETA, min=1e-10)
        if it in cps:
            out[it] = tau_at(A, tau_g, cy, cx)
    return out


def emitter_maps(sig_px_gen):
    """Emitter map for the nominal data-generation PSF."""
    psf_gen = gaussian_psf(sig_px_gen)
    H = W = FIELD
    cy = cx = FIELD // 2
    h_map = np.zeros((H, W))
    r = psf_gen.shape[0] // 2
    h_map[cy - r:cy + r + 1, cx - r:cx + r + 1] = psf_gen
    return h_map, cy, cx


def main():
    backend = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"
    log(f"Backend: {backend}   dtype: {DT}")
    os.makedirs(OUT_DIR, exist_ok=True)

    t = np.arange(N_T) * DT_NS
    sig_px = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_NM

    # ---- nominal geometry (reconstruction side, and photon normalization) ----
    h_nom, cy, cx = emitter_maps(sig_px)
    h0 = float(h_nom[cy, cx])
    FG = h_nom >= FG_FRAC * h0
    hbar = float(h_nom[FG].mean())     # photon budget normalizer, FIXED across
                                       # conditions so only model error varies

    tau_grid = np.round(np.arange(TAU_MIN, TAU_MAX + 1e-9, TAU_STEP), 4)
    K = len(tau_grid)
    D_np = decay(tau_grid, t, IRF_T0_NS, IRF_SIG_NS)
    d_true = decay(TAU_TRUE, t, IRF_T0_NS, IRF_SIG_NS)[0]
    D = torch.as_tensor(D_np, dtype=DT, device=DEV).contiguous()
    Dt = D.t().contiguous()
    Dsum = D.sum(1).reshape(1, K, 1, 1)
    tau_g = torch.as_tensor(tau_grid, dtype=DT, device=DEV).reshape(1, K)

    log(f"tau_true = {TAU_TRUE} ns, checkpoints {ITER_CHECKPOINTS}")
    log(f"photon levels {MEAN_PH_LEVELS}, {N_MC} MC realizations, "
        f"conditions {list(PSF_SCALE_CONDITIONS)}")

    results = []
    t0w = time.time()

    for cond_name, scale in PSF_SCALE_CONDITIONS.items():
        # Data are always generated with the nominal PSF. Only the PSF supplied
        # to the reconstruction is perturbed, matching Supplementary Sec. S7.
        shape_t = torch.as_tensor(h_nom[:, :, None] * d_true[None, None, :],
                                  dtype=DT, device=DEV)
        psf_recon = gaussian_psf(sig_px * scale)
        o = make_otf(psf_recon, (FIELD, FIELD))
        C = _op(torch.ones((1, K, FIELD, FIELD), dtype=DT, device=DEV),
                o, False) * Dsum

        for mp in MEAN_PH_LEVELS:
            N_tot = mp / hbar                      # fixed budget across conditions
            mu = N_tot * shape_t + BG_PER_BIN

            # same seed per (condition, level): iteration counts share
            # realizations by construction; conditions share the random stream
            gen = torch.Generator(device=DEV)
            gen.manual_seed(SEED + 7919 * mp)

            taus = {it: [] for it in ITER_CHECKPOINTS}
            done = 0
            while done < N_MC:
                b = min(MC_CHUNK, N_MC - done)
                Y = torch.poisson(mu.expand(b, FIELD, FIELD, N_T).contiguous(),
                                  generator=gen)
                snap = joint_checkpoint_taus(Y, D, Dt, o, BG_PER_BIN, C,
                                             ITER_CHECKPOINTS, tau_g, cy, cx)
                for it, v in snap.items():
                    taus[it].append(v)
                del Y
                done += b
            torch.cuda.empty_cache()

            for it in ITER_CHECKPOINTS:
                v = np.concatenate(taus[it])
                v = v[v > 1e-6]                    # count, don't hide, empty fits
                bias = float(v.mean() - TAU_TRUE)
                sd = float(v.std(ddof=1))
                rmse = float(np.sqrt(bias ** 2 + sd ** 2))
                # 1-sigma MC error of the SD (Gaussian approx): sd/sqrt(2(n-1))
                sd_se = sd / np.sqrt(2 * (len(v) - 1))
                results.append(dict(condition=cond_name, psf_scale=scale,
                                    mean_ph=mp, n_iter=it, n_ok=int(len(v)),
                                    bias=bias, sd=sd, rmse=rmse, sd_se=sd_se))
            r_lvl = [r for r in results
                     if r["condition"] == cond_name and r["mean_ph"] == mp]
            best = min(r_lvl, key=lambda r: r["rmse"])
            log(f"  [{cond_name}] mean_ph={mp}: argmin RMSE at n_iter="
                f"{best['n_iter']} (rmse {best['rmse']:.3f}, "
                f"bias {best['bias']:+.3f}, sd {best['sd']:.3f})")

    log(f"\n({time.time() - t0w:.0f}s total)")

    # ---------------- helpers for the selection figure ----------------
    def grid_of(cond, key="rmse"):
        """(n_iter x mean_ph) matrix for one condition."""
        M = np.full((len(ITER_CHECKPOINTS), len(MEAN_PH_LEVELS)), np.nan)
        for r in results:
            if r["condition"] != cond:
                continue
            i = ITER_CHECKPOINTS.index(r["n_iter"])
            j = MEAN_PH_LEVELS.index(r["mean_ph"])
            M[i, j] = r[key]
        return M

    def range_curves(M):
        """Raw mean over the photon range, and normalized excess ratio.

        The raw mean is dominated by the low-photon end (largest RMSE), which
        matches the paper's regime of interest; the normalized curve
        mean_p[ RMSE(N,p) / min_N RMSE(N,p) ] is scale-free and checks that
        the choice is not an artifact of that weighting.
        """
        raw = M.mean(axis=1)
        ratio = (M / M.min(axis=0, keepdims=True)).mean(axis=1)
        return raw, ratio

    # ---------------- selection figure: (a) heatmap  (b) range-average ------
    colors = {"nominal": "#000000", "psf+10%": "#0072B2", "psf-10%": "#E69F00"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) heatmap, nominal condition: x = photons, y = iterations, color = RMSE
    Mn = grid_of("nominal")
    pc = axA.pcolormesh(MEAN_PH_LEVELS, ITER_CHECKPOINTS, Mn,
                        shading="nearest", cmap="viridis")
    axA.set_yscale("log")
    axA.set_xlabel("Mean photons per pixel", fontsize=12)
    axA.set_ylabel("Iterations", fontsize=12)
    axA.set_title("(a) RMSE vs true lifetime (nominal)", fontsize=12)
    cb = fig.colorbar(pc, ax=axA)
    cb.set_label("RMSE (ns)", fontsize=11)
    # ridge of per-photon optima
    for j, mp in enumerate(MEAN_PH_LEVELS):
        i = int(np.nanargmin(Mn[:, j]))
        axA.plot(mp, ITER_CHECKPOINTS[i], "w*", ms=9, mec="k", mew=0.5)

    # (b) range-averaged RMSE vs iteration, all conditions
    sel = {}
    for cond in PSF_SCALE_CONDITIONS:
        raw, ratio = range_curves(grid_of(cond))
        axB.plot(ITER_CHECKPOINTS, raw, "o-", ms=5, lw=1.8,
                 color=colors.get(cond, None), label=cond)
        sel[cond] = dict(
            argmin_raw=int(ITER_CHECKPOINTS[int(np.nanargmin(raw))]),
            argmin_ratio=int(ITER_CHECKPOINTS[int(np.nanargmin(ratio))]),
            raw=[float(v) for v in raw], ratio=[float(v) for v in ratio])
        axB.axvline(sel[cond]["argmin_raw"], color=colors.get(cond, None),
                    ls=":", lw=1, alpha=0.7)
    axB.set_xscale("log")
    axB.set_xlabel("Iterations", fontsize=12)
    axB.set_ylabel("RMSE averaged over 1-15 photons/px (ns)", fontsize=12)
    axB.set_title("(b) range-averaged RMSE", fontsize=12)
    axB.grid(alpha=0.3, which="both")
    axB.legend(fontsize=9, frameon=False)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "iteration_selection.png"), dpi=220)
    plt.close(fig)

    # ---------------- serialize ----------------
    payload = dict(tau_true=TAU_TRUE, n_mc=N_MC, eta=ETA,
                   checkpoints=ITER_CHECKPOINTS, mean_ph_levels=MEAN_PH_LEVELS,
                   conditions=PSF_SCALE_CONDITIONS, selection=sel,
                   results=results)
    with open(os.path.join(OUT_DIR, "iter_optimization_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ---------------- summary ----------------
    with open(os.path.join(OUT_DIR, "iter_optimization_summary.txt"), "w") as f:
        f.write(f"Iteration selection on single-emitter simulation, "
                f"tau_true = {TAU_TRUE} ns, {N_MC} MC\n")
        f.write("metric: per-level RMSE^2 = bias^2 + SD^2 at the emitter pixel\n")
        f.write("selection: argmin over iterations of the RMSE averaged across "
                "the paper's photon range (1-15 ph/px);\n")
        f.write("cross-checked with the normalized excess-ratio average "
                "mean_p[RMSE(N,p)/min_N RMSE(N,p)]\n")
        f.write("data PSF perturbed per condition; reconstruction PSF nominal; "
                "photon budget fixed\n\n")
        for cond, s in sel.items():
            f.write(f"[{cond}] argmin range-mean RMSE: n_iter="
                    f"{s['argmin_raw']}   argmin normalized ratio: n_iter="
                    f"{s['argmin_ratio']}\n")
            f.write(f"{'n_iter':>8}{'mean_rmse':>11}{'mean_ratio':>12}\n")
            for it, rv, tv in zip(ITER_CHECKPOINTS, s["raw"], s["ratio"]):
                f.write(f"{it:>8}{rv:>11.4f}{tv:>12.4f}\n")
            f.write("\n")
        f.write("full grid:\n")
        f.write(f"{'condition':>10}{'mean_ph':>9}{'n_iter':>8}{'bias':>9}"
                f"{'sd':>8}{'rmse':>8}{'n_ok':>6}\n")
        for r in results:
            f.write(f"{r['condition']:>10}{r['mean_ph']:>9}{r['n_iter']:>8}"
                    f"{r['bias']:>9.3f}{r['sd']:>8.3f}{r['rmse']:>8.3f}"
                    f"{r['n_ok']:>6}\n")

    log(f"-> {OUT_DIR}/iteration_selection.png")
    log(f"-> {OUT_DIR}/iter_optimization_results.json / _summary.txt")


if __name__ == "__main__":
    main()

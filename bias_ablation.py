"""
Where does the lifetime bias come from: the dictionary, or early stopping?

The sensitivity sweep left an unresolved attribution. At SBR 10, 1 photon/pixel:

    oracle     bias = -0.014 ns     continuous tau, no dictionary
    pixel-wise bias = +0.157 ns     dictionary, 1000 iterations
    proposed   bias = +0.173 ns     dictionary,   50 iterations

The pixel-wise estimator runs to convergence and still carries essentially the
same bias as the 20-iteration joint estimator, while the only estimator without
a dictionary carries none. That points at the dictionary rather than at early
stopping, but the two were never varied independently. This does that.

    axis 1  TAU_STEP        0.4 (pipeline), 0.2, 0.1, 0.05
    axis 2  JOINT_N_ITER    50 (pipeline), 100
    control oracle          dictionary-free; its bias should not move at all

Read it as:
    bias falls along axis 1, flat along axis 2   -> dictionary discretisation
    bias falls along axis 2, flat along axis 1   -> early stopping
    falls along both                             -> both contribute, and the
                                                    table gives their relative size

An on-grid confound to be aware of
----------------------------------
TAU_TRUE = 2.0 does NOT sit on the 0.4 grid (1.0, 1.4, 1.8, 2.2, ...) but DOES
sit on the 0.2, 0.1 and 0.05 grids. A drop in bias between 0.4 and 0.2 could
therefore be the true value landing on a grid point rather than the grid
getting finer. The script defaults to running a second true lifetime, 2.1 ns,
which is off-grid for 0.4, 0.2 and 0.05 and on-grid only for 0.1, so the two
rows disagree about which steps are "lucky". If the trend survives in both,
it is the grid spacing; if it only appears where the true value is on-grid,
the bias is discretisation snapping and a finer dictionary is a real fix but
for a different reason than assumed.

Common random numbers: the sweep seeds only on the photon level, so every cell
at a given true lifetime sees identical Poisson realisations. The comparison
across the table is exactly paired.

Usage
-----
    python bias_ablation.py                                   # defaults
    python bias_ablation.py --steps 0.4,0.2,0.1 --iters 20,100
    python bias_ablation.py --taus 2.0 --n-mc 960             # tighter, one tau
    python bias_ablation.py --chunk 32                        # if VRAM is tight

Cost scales with the dictionary size K = (TAU_MAX - TAU_MIN)/step + 1, which is
11 / 21 / 41 / 81 for the four default steps, and the pixel-wise MLE runs 1000
iterations at every K. Start with --steps 0.4,0.1 if you want a quick answer.
"""

import argparse
import json

import numpy as np
import torch

import sensitivity_sweep_pipeline as sp


NOMINAL = dict(cid=0, tags=["ablation"], sbr=10.0, psf_err=0.0, bkg_err=0.0)


def build_shared(tau_step, tau_true):
    """Rebuild everything that depends on the dictionary or the true lifetime."""
    t = np.arange(sp.N_T) * sp.DT_NS
    sig = (sp.PSF_FWHM_NM / 2.3548200450309493) / sp.PIXEL_NM
    psf = sp.gaussian_psf(sig)
    H = W = sp.FIELD
    cy = cx = sp.FIELD // 2
    h_map = np.zeros((H, W))
    r = psf.shape[0] // 2
    h_map[cy - r:cy + r + 1, cx - r:cx + r + 1] = psf
    h0 = float(h_map[cy, cx])
    FG = h_map >= sp.FG_FRAC * h0
    hbar = float(h_map[FG].mean())

    tau_grid = np.round(np.arange(sp.TAU_MIN, sp.TAU_MAX + 1e-9, tau_step), 6)
    D_np = sp.decay(tau_grid, t, sp.IRF_T0_NS, sp.IRF_SIG_NS)
    d_true = sp.decay(tau_true, t, sp.IRF_T0_NS, sp.IRF_SIG_NS)[0]
    D = torch.as_tensor(D_np, dtype=sp.DT, device=sp.DEV).contiguous()

    on_grid = bool(np.any(np.abs(tau_grid - tau_true) < 1e-9))
    shared = dict(
        t=t, D=D, Dt=D.t().contiguous(),
        Dsum=D.sum(1).reshape(1, len(tau_grid), 1, 1),
        tau_g=torch.as_tensor(tau_grid, dtype=sp.DT, device=sp.DEV).reshape(1, len(tau_grid)),
        sig_true=sig, h_map=h_map, FG=FG, hbar=hbar, h0=h0, h_flat=h_map.ravel(),
        shape=torch.as_tensor(h_map[:, :, None] * d_true[None, None, :],
                              dtype=sp.DT, device=sp.DEV))
    return shared, len(tau_grid), on_grid


# ---------------------------------------------------------------------------
# Efficient scan engine
# ---------------------------------------------------------------------------
# Two savings, both of which matter when JOINT_N_ITER is the axis being swept:
#   1. The pixel-wise MLE (1000 iterations, the dominant cost) and the oracle do
#      not depend on JOINT_N_ITER at all, so they are computed once per
#      (tau_true, tau_step) cell and reused.
#   2. The joint iterations are nested: iterating to 50 passes through 10, 15,
#      20 and 30 on the way. The update below is a checkpointed copy of
#      sp.joint_batch that reads out tau at each requested count in a single
#      pass, so an n-value scan costs the same as one value.
# The update must stay identical to sp.joint_batch; verify_checkpoints() below
# asserts that at startup rather than trusting the copy.

def setup_cell(cond, mean_ph, shared):
    """Mirrors the per-condition setup inside sp.run_cell."""
    H = W = sp.FIELD
    cy = cx = sp.FIELD // 2
    sig_a = shared["sig_true"] * (1.0 + cond["psf_err"])
    o = sp.make_otf(sp.gaussian_psf(sig_a), (H, W))
    C = sp._op(torch.ones((1, shared["D"].shape[0], H, W), dtype=sp.DT, device=sp.DEV),
               o, False) * shared["Dsum"]
    yy, xx = np.mgrid[0:H, 0:W]
    ROI = np.hypot(yy - cy, xx - cx) <= sp.ROI_SIGMA * sig_a
    bg_true = mean_ph / (cond["sbr"] * sp.N_T)
    bg_a = bg_true * (1.0 + cond["bkg_err"])
    N_tot = mean_ph / shared["hbar"]
    return dict(o=o, C=C, ROI=ROI, bg_true=bg_true, bg_a=bg_a, N_tot=N_tot,
                sig_a=sig_a, cy=cy, cx=cx,
                mu=N_tot * shared["shape"] + bg_true)


def joint_checkpoints(Y, D, Dt, o, bg, C, want, tau_g, cy, cx):
    """sp.joint_batch, reading out tau at every count in `want` in one pass."""
    B, H, W, T = Y.shape
    K = D.shape[0]
    A = torch.full((B, K, H, W), 1e-2, dtype=sp.DT, device=sp.DEV)
    want = sorted(set(want))
    out = {}
    for it in range(1, want[-1] + 1):
        lam = sp.lam_of(sp._op(A, o, False), D) + bg
        U = sp._op(sp.back_of(Y / (lam + sp.EPS), Dt), o, True)
        A = torch.clamp(A * torch.clamp(U / (C + sp.EPS), min=0.0) ** sp.ETA, min=1e-10)
        if it in want:
            out[it] = sp.tau_at(A, tau_g, cy, cx)
    return out


def verify_checkpoints(shared, cell, n_iter=7, B=4):
    """Assert the checkpointed loop reproduces sp.joint_batch exactly."""
    gen = torch.Generator(device=sp.DEV); gen.manual_seed(12345)
    H = W = sp.FIELD
    Y = torch.poisson(cell["mu"].expand(B, H, W, sp.N_T).contiguous(), generator=gen)
    a = joint_checkpoints(Y, shared["D"], shared["Dt"], cell["o"], cell["bg_a"],
                          cell["C"], [n_iter], shared["tau_g"], cell["cy"], cell["cx"])[n_iter]
    b = sp.tau_at(sp.joint_batch(Y, shared["D"], shared["Dt"], cell["o"], cell["bg_a"],
                                 cell["C"], n_iter), shared["tau_g"], cell["cy"], cell["cx"])
    d = float(np.max(np.abs(a - b)))
    print(f"[check] checkpointed joint update matches sp.joint_batch to {d:.2e} ns")
    return d


def run_scan(cond, mean_ph, shared, iters, n_mc, chunk, do_oracle=True):
    """One (tau_true, tau_step) cell, all iteration counts, common random numbers."""
    cell = setup_cell(cond, mean_ph, shared)
    H = W = sp.FIELD
    gen = torch.Generator(device=sp.DEV)
    gen.manual_seed(sp.SEED + int(round(mean_ph * 10)))   # same CRN as the sweep

    acc = {f"proposed@{i}": [] for i in iters}
    acc["pixelwise"] = []
    if do_oracle:
        acc["pooled"] = []

    done = 0
    while done < n_mc:
        b = min(chunk, n_mc - done)
        Y = torch.poisson(cell["mu"].expand(b, H, W, sp.N_T).contiguous(), generator=gen)
        acc["pixelwise"].append(sp.tau_at(
            sp.mle_batch(Y, shared["D"], shared["Dt"], shared["Dsum"],
                         cell["bg_a"], sp.MLE_N_ITER), shared["tau_g"],
            cell["cy"], cell["cx"]))
        ck = joint_checkpoints(Y, shared["D"], shared["Dt"], cell["o"], cell["bg_a"],
                               cell["C"], iters, shared["tau_g"], cell["cy"], cell["cx"])
        for i in iters:
            acc[f"proposed@{i}"].append(ck[i])
        if do_oracle:
            Yn = Y.double().cpu().numpy()
            n_roi = int(cell["ROI"].sum())
            acc["pooled"].append(np.array(
                [sp.pooled_mle_tau(Yn[i][cell["ROI"]].sum(axis=0), shared["t"],
                                   cell["bg_a"] * n_roi) for i in range(b)]))
        del Y
        done += b
    if sp.DEV.type == "cuda":
        torch.cuda.empty_cache()

    per = n_mc // sp.N_BLOCK
    stats = {}
    for m, chunks in acc.items():
        v = np.concatenate(chunks)
        bl = []
        for k in range(sp.N_BLOCK):
            x = v[k * per:(k + 1) * per]
            x = x[x > 1e-6]
            if x.size < 3:
                continue
            e = x - sp.TAU_TRUE
            bl.append((float(x.std(ddof=1)), float(e.mean()),
                       float(np.sqrt((e ** 2).mean()))))
        bl = np.array(bl)
        stats[m] = dict(sd=float(bl[:, 0].mean()), bias=float(bl[:, 1].mean()),
                        rmse=float(bl[:, 2].mean()),
                        sd_se=float(bl[:, 0].std(ddof=1) / np.sqrt(len(bl))),
                        bias_se=float(bl[:, 1].std(ddof=1) / np.sqrt(len(bl))),
                        rmse_se=float(bl[:, 2].std(ddof=1) / np.sqrt(len(bl))))
    crb_px = float(np.sqrt(sp.crb_pixel(sp.TAU_TRUE, cell["N_tot"] * shared["h0"],
                                        shared["t"], sp.IRF_T0_NS, sp.IRF_SIG_NS,
                                        cell["bg_true"])))
    crb_pl = float(np.sqrt(sp.crb_pooled(sp.TAU_TRUE, cell["N_tot"], shared["h_flat"],
                                         shared["t"], sp.IRF_T0_NS, sp.IRF_SIG_NS,
                                         cell["bg_true"])))
    return stats, dict(crb_pixel=crb_px, crb_pooled=crb_pl, bg_per_bin=cell["bg_true"],
                       roi_px=int(cell["ROI"].sum()), sigma_assumed=cell["sig_a"])


def cell_stats(rows):
    out = {}
    for m in set(r["method"] for r in rows):
        sel = [r for r in rows if r["method"] == m]
        out[m] = dict(
            bias=float(np.mean([r["bias_gt"] for r in sel])),
            sd=float(np.mean([r["sd_gt"] for r in sel])),
            rmse=float(np.mean([r["rmse_gt"] for r in sel])),
            bias_se=float(np.std([r["bias_gt"] for r in sel], ddof=1) / np.sqrt(len(sel))),
            sd_se=float(np.std([r["sd_gt"] for r in sel], ddof=1) / np.sqrt(len(sel))),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="0.4,0.2,0.1,0.05")
    ap.add_argument("--iters", default="50,100")
    ap.add_argument("--taus", default="2.0,2.1")
    ap.add_argument("--photons", type=float, default=1.0)
    ap.add_argument("--n-mc", type=int, default=sp.N_MC_DEFAULT)
    ap.add_argument("--chunk", type=int, default=sp.MC_CHUNK)
    ap.add_argument("--mle-iters", type=int, default=None,
                    help="override MLE_N_ITER (testing only; the pipeline uses 1000)")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="bias_ablation.json")
    a = ap.parse_args()

    steps = [float(x) for x in a.steps.split(",")]
    iters = sorted(int(x) for x in a.iters.split(","))
    taus = [float(x) for x in a.taus.split(",")]
    n_mc = (a.n_mc // sp.N_BLOCK) * sp.N_BLOCK
    if n_mc // sp.N_BLOCK < 8:
        raise SystemExit(f"--n-mc must be at least {8 * sp.N_BLOCK}")
    if a.mle_iters:
        sp.MLE_N_ITER = a.mle_iters
        print(f"[warn] MLE_N_ITER overridden to {sp.MLE_N_ITER} (pipeline value is 1000)")

    results = []
    checked = False
    for tau_true in taus:
        sp.TAU_TRUE = tau_true
        print(f"\n=== true lifetime {tau_true} ns | SBR {NOMINAL['sbr']:.0f}, matched PSF "
              f"and background, {a.photons:g} ph/px, {n_mc} MC ===")
        for step in steps:
            shared, K, on_grid = build_shared(step, tau_true)
            if not checked:
                verify_checkpoints(shared, setup_cell(NOMINAL, a.photons, shared))
                checked = True
            st, meta = run_scan(NOMINAL, a.photons, shared, iters, n_mc, a.chunk,
                                not a.no_oracle)
            print(f"  step {step:g}  K={K}  true lifetime is "
                  f"{'ON' if on_grid else 'OFF'} the grid   "
                  f"[pixel-wise bias {st['pixelwise']['bias']:+.4f}, "
                  f"sd {st['pixelwise']['sd']:.4f}"
                  + (f" | oracle bias {st['pooled']['bias']:+.4f}, "
                     f"sd {st['pooled']['sd']:.4f}]" if "pooled" in st else "]"))
            print(f"  {'n_iter':>7} {'bias':>9} {'+/-':>7} {'sd':>8} {'+/-':>7} "
                  f"{'rmse':>8} {'+/-':>7}   {'sd/CRBpool':>10}")
            best = min(iters, key=lambda i: st[f"proposed@{i}"]["rmse"])
            interior = len(iters) > 2 and best not in (iters[0], iters[-1])
            for i in iters:
                v = st[f"proposed@{i}"]
                mark = ("  <- min RMSE" + ("" if interior else " (AT RANGE EDGE)")) \
                    if i == best else ""
                print(f"  {i:>7d} {v['bias']:>+9.4f} {v['bias_se']:>7.4f} "
                      f"{v['sd']:>8.4f} {v['sd_se']:>7.4f} {v['rmse']:>8.4f} "
                      f"{v['rmse_se']:>7.4f}   {v['sd'] / meta['crb_pooled']:>10.3f}{mark}")
            results.append(dict(tau_true=tau_true, tau_step=step, K=K, on_grid=on_grid,
                                photons=a.photons, iters=iters, stats=st, meta=meta,
                                rmse_min_iter=best, rmse_min_is_interior=interior))
            if not interior:
                print(f"  [note] the RMSE minimum sits at the edge of the tested range "
                      f"{iters}, so this says only that {best} beats the other value(s) "
                      f"tried, not that it is optimal. Bracket it, e.g. "
                      f"--iters 10,15,20,30,50.")

    with open(a.out, "w") as fh:
        json.dump(dict(photons=a.photons, n_mc=n_mc, mle_n_iter=sp.MLE_N_ITER,
                       condition=NOMINAL, results=results), fh, indent=2)
    print(f"\n-> {a.out}")

    if not a.no_plot and len(iters) > 2:
        try:
            plot_iter_scan(results, iters)
        except Exception as exc:
            print(f"[warn] plot skipped: {exc}")


def plot_iter_scan(results, iters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

    sub = [r for r in results if r["tau_step"] == results[0]["tau_step"]
           and r["tau_true"] == results[0]["tau_true"]]
    r = sub[0]
    st = r["stats"]
    b = [abs(st[f"proposed@{i}"]["bias"]) for i in iters]
    sd = [st[f"proposed@{i}"]["sd"] for i in iters]
    rm = [st[f"proposed@{i}"]["rmse"] for i in iters]
    se = [st[f"proposed@{i}"]["rmse_se"] for i in iters]

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(iters, b, "o-", color="#B23A48", label="|bias|")
    ax.plot(iters, sd, "s-", color="#4A6FA5", label="s.d.")
    ax.errorbar(iters, rm, yerr=se, fmt="D-", color="#0B6E4F", capsize=1.5,
                elinewidth=0.6, label="RMSE")
    if "pooled" in st:
        ax.axhline(st["pooled"]["sd"], color="0.55", lw=0.7, ls=(0, (4, 2)))
        ax.annotate("oracle s.d.", xy=(iters[-1], st["pooled"]["sd"]),
                    xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=6.5, color="0.4")
    k = r["rmse_min_iter"]
    ax.plot([k], [st[f"proposed@{k}"]["rmse"]], "o", ms=9, mfc="none",
            mec="#0B6E4F", mew=1.0)
    ax.set_xscale("log")
    ax.set_xticks(iters); ax.set_xticklabels([str(i) for i in iters])
    ax.set_xlabel("joint iterations")
    ax.set_ylabel("lifetime error (ns)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"$\\tau$ = {r['tau_true']} ns, dictionary step "
                 f"{r['tau_step']:g} ns, {r['photons']:g} ph/px", fontsize=7.5)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig("bias_ablation_iters.pdf")
    fig.savefig("bias_ablation_iters.png", dpi=400)
    print("-> bias_ablation_iters.pdf / .png")


if __name__ == "__main__":
    main()

"""
FLIM low-photon benchmark simulation (FPFLI-matched temporal sampling)
=====================================
Provides the common 256-bin forward model used to compare two-lifetime
(2 ns / 4 ns) bead maps with the official FPFLI model.  The temporal grid
matches the published/pretrained FPFLI input rather than interpolating the
old 64-bin benchmark after acquisition.

  1. Pixel-wise Poisson MLE      (no spatial coupling; "no-denoise gold standard")
  2. PCA-denoise + NNLS unmix    (linear, training-free denoising, proxy for NC-PCA)
  3. CNN proxy (MLP on patches)  (data-driven, supervised; lightweight stand-in for FLI-Net
                                   since no GPU/torch is available in this offline sandbox)
  4. Our method                  (joint spatial-temporal Poisson MLE, PSF-coupled,
                                   multiplicative Richardson-Lucy/EM-style update)

Forward model
-------------
Lambda(r,t) = sum_k [h * A_k](r) * D_k(t) + B(r,t)
Y(r,t) ~ Poisson(Lambda(r,t))

Units: spatial in pixels (pixel_size_nm), time in ns.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import nnls

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------
# 1. Physical / imaging parameters
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 1. Physical / imaging parameters
# ----------------------------------------------------------------------
PIXEL_SIZE_NM = 50
PSF_FWHM_NM = 200.0
PSF_SIGMA_PX = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_SIZE_NM  # 1.699 px

IMAGE_SIZE = 100           # 100x100 px -> 5.0 um x 5.0 um FOV at 50 nm
N_TIME_BINS = 256
# Temporal axis matched to the experimental acquisitions and to
# single_bead_crb_mc.py, rather than to the axis FPFLI was trained on
# (0.039 ns, 9.98 ns window). 256 bins x 0.0977 ns = 25.0 ns, i.e. 8.3
# lifetimes for the longest-lived species.
DT_NS = 0.0977
IRF_FWHM_NS = 0.150        # measured IRFs span 116-157 ps FWHM
IRF_PEAK_BIN = 5.1177      # t0 = 0.500 ns, as in the single-emitter study
TAUS_NS = [2, 4]       # two lifetime species
# Flat dark-count rate per (pixel, time bin). 1e-4/bin = 0.026 photons per
# pixel over the window. This is already conservative: the 114 pre-pulse bins
# of the microtubule acquisition contain no counts at all across the whole
# field, bounding the true background below 2.4e-6 per pixel per bin.
BACKGROUND_RATE = 1e-4
from scipy.special import erf
EPS = 1e-9


# ----------------------------------------------------------------------
# 2. Build PSF kernel and temporal decay bases
# ----------------------------------------------------------------------
def _gaussian_psf(sigma_px):
    r = max(int(np.ceil(4 * sigma_px)), 1)
    a = np.arange(-r, r + 1)
    w = 0.5 * (erf((a + 0.5) / (np.sqrt(2) * sigma_px))
               - erf((a - 0.5) / (np.sqrt(2) * sigma_px)))
    p = np.outer(w, w)
    return p / p.sum()


def make_decay_bases(taus_ns, n_bins, dt_ns, irf_fwhm_ns):
    """D_k(t) = analytic convolution of a Gaussian IRF with exp(-t/tau),
    each normalized to unit area.

    The analytic form is used rather than a discrete convolution of a sampled
    Gaussian: at the experimental sampling the IRF is only 0.65 bins wide, and
    sampling it at bin centres then convolving numerically shifts the peak by
    one bin and changes the profile appreciably for the shortest lifetime
    species. The single-emitter study uses this same expression, so the two
    simulations now share one forward model exactly.
    """
    from scipy.special import erfc
    t = np.arange(n_bins) * dt_ns
    sig = irf_fwhm_ns / 2.3548200450309493
    t0 = IRF_PEAK_BIN * dt_ns
    tau = np.atleast_1d(np.asarray(taus_ns, dtype=float))[:, None]
    tt = t[None, :]
    lam = 1.0 / tau
    arg = (t0 + lam * sig ** 2 - tt) / (np.sqrt(2) * sig)
    ex = np.clip(0.5 * lam ** 2 * sig ** 2 - lam * (tt - t0), -700, 700)
    d = np.clip(0.5 * lam * np.exp(ex) * erfc(arg), 0.0, None)
    s = d.sum(axis=1, keepdims=True)
    return d / np.where(s > 0, s, 1.0)
PSF = _gaussian_psf(PSF_SIGMA_PX)
PSF_PEAK_HEIGHT = PSF.max()  # normalized PSF peak height (sum(PSF)=1)


def photons_per_bead_to_mean_per_pixel(photons_per_bead):
    """Converts the internal simulation unit (total expected photons
    contributed by one bead, integrated over its full PSF-blurred spatial
    footprint and full decay) into the conventional FLIM reporting unit:
    the mean photon count at the bead's peak (brightest) pixel, which is
    what actually determines pixel-wise fitting SNR. "photons per bead" is
    an internal simulation parameter; "photons per pixel" (peak-pixel
    count) is what FLIM readers expect and compare against literature
    values (e.g., FLIMngo's "10-100 photons per pixel" operating range)."""
    return np.asarray(photons_per_bead) * PSF_PEAK_HEIGHT
D_BASES = make_decay_bases(TAUS_NS, N_TIME_BINS, DT_NS, IRF_FWHM_NS)  # (K, T)
K_SPECIES = len(TAUS_NS)


def conv2d_same(img, kernel):
    return fftconvolve(img, kernel, mode="same")


# ----------------------------------------------------------------------
# 3. Ground-truth scene generation
# ----------------------------------------------------------------------
def generate_scene(n_beads, photons_per_bead, rng, margin=8, min_sep_px=1, n_species0=None):
    """Randomly place beads on the pixel grid, assigning lifetime species
    0 (2 ns) or 1 (4 ns) with a fixed, balanced count rather than an
    independent 50/50 coin flip per bead (which can by chance produce an
    imbalanced split, e.g. 11 vs. 4 out of 15). By default, n_species0 is
    set to ceil(n_beads/2), so n_beads=15 gives an 8/7 split; which
    specific beads get which species is still randomized via rng."""
    if n_species0 is None:
        n_species0 = -(-n_beads // 2)  # ceil(n_beads/2) -> 8 for n_beads=15
    species_pool = [0] * n_species0 + [1] * (n_beads - n_species0)

    A_true = np.zeros((K_SPECIES, IMAGE_SIZE, IMAGE_SIZE))
    coords = []
    tries = 0
    while len(coords) < n_beads and tries < 10000:
        tries += 1
        x = rng.integers(margin, IMAGE_SIZE - margin)
        y = rng.integers(margin, IMAGE_SIZE - margin)
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_sep_px ** 2 for cx, cy in coords):
            coords.append((x, y))
    # Shuffle which bead (position) gets which species, but the COUNT per
    # species is fixed at n_species0 / (n_beads - n_species0).
    species_assignment = rng.permutation(species_pool[:len(coords)])
    for (x, y), k in zip(coords, species_assignment):
        A_true[k, y, x] = photons_per_bead
    return A_true, coords


def forward_model(A_true):
    """Compute noiseless Lambda(r,t) cube, shape (H, W, T)."""
    H, W = A_true.shape[1:]
    Lambda = np.zeros((H, W, N_TIME_BINS))
    for k in range(K_SPECIES):
        blurred = conv2d_same(A_true[k], PSF)
        Lambda += blurred[:, :, None] * D_BASES[k][None, None, :]
    Lambda += BACKGROUND_RATE
    return Lambda


def simulate_measurement(A_true, rng):
    Lambda = forward_model(A_true)
    Y = rng.poisson(Lambda).astype(np.float64)
    return Y, Lambda


# ----------------------------------------------------------------------
# 4. Method 1: pixel-wise Poisson MLE (no spatial coupling)
# ----------------------------------------------------------------------
def pixelwise_poisson_mle(Y, n_iter=150, eta=1.0):
    H, W, T = Y.shape
    A = np.ones((K_SPECIES, H, W)) * (Y.sum() / (H * W * K_SPECIES) + EPS)
    for _ in range(n_iter):
        Lambda = np.einsum('khw,kt->hwt', A, D_BASES) + BACKGROUND_RATE
        R = Y / (Lambda + EPS)
        for k in range(K_SPECIES):
            U_k = np.tensordot(R, D_BASES[k], axes=([2], [0]))  # (H,W)
            C_k = D_BASES[k].sum()  # scalar, ~1
            A[k] *= (U_k / (C_k + EPS)) ** eta
    return A


# ----------------------------------------------------------------------
# 2b. Method: fit-free phasor analysis (Digman et al. 2008), 80 MHz
#
# Standard, widely-used fit-free lifetime estimator: per-pixel phasor
# coordinates (G, S) at a chosen modulation frequency, converted to a
# single-exponential lifetime via tau = S / (omega * G). Included as an
# additional baseline because it is the most commonly used fit-free method
# in the field, distinct from the pixel-wise Poisson MLE / PCA baselines.
# ----------------------------------------------------------------------
def phasor_lifetime_map(Y, dt_ns=DT_NS, freq_mhz=80.0):
    H, W, T = Y.shape
    t = np.arange(T) * dt_ns
    omega = 2 * np.pi * freq_mhz * 1e-3  # rad/ns (freq_mhz * 1e6 Hz -> * 1e-9 s/ns)
    cos_t = np.cos(omega * t)
    sin_t = np.sin(omega * t)

    total = Y.sum(axis=2)
    G = np.divide(np.tensordot(Y, cos_t, axes=([2], [0])), total,
                  out=np.zeros((H, W)), where=total > EPS)
    S = np.divide(np.tensordot(Y, sin_t, axes=([2], [0])), total,
                  out=np.zeros((H, W)), where=total > EPS)

    tau_map = np.divide(S, omega * G, out=np.full((H, W), np.nan), where=np.abs(G) > 1e-6)
    return tau_map, G, S


def phasor_lifetime_at_points(Y, coords, dt_ns=DT_NS, freq_mhz=80.0):
    tau_map, G, S = phasor_lifetime_map(Y, dt_ns=dt_ns, freq_mhz=freq_mhz)
    return np.array([tau_map[y, x] for (x, y) in coords])
#
# Faithful re-implementation of the "NC-PCA" (noise-corrected PCA) algorithm
# from Soltani, Paulson et al., "Denoising of fluorescence lifetime imaging
# data via principal component analysis," Sci. Rep. 15, 40258 (2025),
# following their released code (Zenodo 10.5281/zenodo.14895223, NCPCA.py,
# Single_Image_PCA.single_image_pca_time_var). Unlike a generic truncated-SVD
# denoise, the key step is a per-time-bin noise-correction (NC) normalization
# before PCA: each time-slice image is divided by sqrt(its own mean) -- an
# approximate Poisson noise scale -- so that PCA is not dominated by
# whichever time bins happen to carry more absolute photon counts; the same
# factor is multiplied back after reconstruction. r=3 components is used,
# matching the value the authors report as optimal in their own study (we do
# not tune r to our known ground truth). NNLS unmixing against the known
# decay bases is then applied exactly as for the previous PCA baseline, to
# keep the downstream comparison unchanged.
# ----------------------------------------------------------------------
# Alternative spatial-deconvolution baselines (RL is what the Proposed
# method's joint update effectively uses for its spatial term; these swap
# in two other classical deconvolution algorithms instead, applied to the
# same per-component abundance maps obtained by pixel-wise temporal NNLS
# unmixing -- i.e., "unmix first, then deconvolve spatially", as a
# two-stage alternative to the Proposed method's single joint update).
# ----------------------------------------------------------------------
def pixelwise_nnls_unmix(Y):
    """Per-pixel temporal NNLS unmixing only, no spatial processing --
    the common starting point for both alternative deconvolution baselines."""
    H, W, T = Y.shape
    basis = D_BASES.T  # (T, K)
    flat = Y.reshape(H * W, T)
    A = np.zeros((K_SPECIES, H * W))
    for i in range(H * W):
        sol, _ = nnls(basis, flat[i])
        A[:, i] = sol
    return A.reshape(K_SPECIES, H, W)


def wiener_deconvolve(img, psf, snr=50.0):
    """Classical frequency-domain Wiener deconvolution of a single 2D image."""
    H, W = img.shape
    psf_padded = np.zeros((H, W))
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf
    psf_padded = np.roll(psf_padded, (-ph // 2, -pw // 2), axis=(0, 1))
    Fpsf = np.fft.fft2(psf_padded)
    Fimg = np.fft.fft2(img)
    wiener_filter = np.conj(Fpsf) / (np.abs(Fpsf) ** 2 + 1.0 / snr)
    out = np.real(np.fft.ifft2(Fimg * wiener_filter))
    return np.clip(out, 0, None)


def wiener_deconv_unmix(Y, psf, snr=50.0):
    A = pixelwise_nnls_unmix(Y)
    return np.stack([wiener_deconvolve(A[k], psf, snr=snr) for k in range(K_SPECIES)])


def tv_deconvolve(img, psf, n_iter=100, lam=0.02, step=None):
    """Isotropic total-variation-regularized deconvolution via gradient
    descent on 0.5||h*x-y||^2 + lam*TV(x), a standard classical alternative
    to Richardson-Lucy for edge-preserving spatial deconvolution."""
    psf_flip = psf[::-1, ::-1]
    x = np.clip(img.copy(), 0, None)
    if step is None:
        step = 1.0 / (np.sum(psf) ** 2 + 1e-6) * 0.5

    def grad_tv(x, eps=1e-6):
        gx = np.roll(x, -1, axis=1) - x
        gy = np.roll(x, -1, axis=0) - x
        norm = np.sqrt(gx ** 2 + gy ** 2 + eps)
        div_x = (gx / norm) - np.roll(gx / norm, 1, axis=1)
        div_y = (gy / norm) - np.roll(gy / norm, 1, axis=0)
        return -(div_x + div_y)

    for _ in range(n_iter):
        resid = conv2d_same(x, psf) - img
        data_grad = conv2d_same(resid, psf_flip)
        x = x - step * (data_grad + lam * grad_tv(x))
        x = np.clip(x, 0, None)
    return x


def tv_deconv_unmix(Y, psf, n_iter=100, lam=0.02):
    A = pixelwise_nnls_unmix(Y)
    return np.stack([tv_deconvolve(A[k], psf, n_iter=n_iter, lam=lam) for k in range(K_SPECIES)])


# ----------------------------------------------------------------------
def ncpca_denoise_unmix(Y, n_components=3):
    H, W, T = Y.shape
    imgs = np.transpose(Y, (2, 0, 1))  # (T, H, W), matching the authors' convention

    norm_factors = np.array([np.sqrt(max(img.mean(), EPS)) for img in imgs])
    imgs_norm = imgs / norm_factors[:, None, None]

    X = imgs_norm.reshape(T, H * W).T  # (H*W, T): pixels x time
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.decomposition import PCA
    r = min(n_components, T, H * W)
    pca = PCA(n_components=r, svd_solver="auto")
    X_pca = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_pca)  # still in NC-normalized scale

    recon_imgs_norm = X_recon.T.reshape(T, H, W)
    recon_imgs = recon_imgs_norm * norm_factors[:, None, None]  # the "NC factor" re-scaling
    recon_imgs = np.clip(recon_imgs, 0, None)

    flat_denoised = np.transpose(recon_imgs, (1, 2, 0)).reshape(H * W, T)

    basis = D_BASES.T  # (T, K)
    A = np.zeros((K_SPECIES, H * W))
    for i in range(H * W):
        sol, _ = nnls(basis, flat_denoised[i])
        A[:, i] = sol
    return A.reshape(K_SPECIES, H, W)


# kept for backward compatibility with any code still calling the old name
def pca_denoise_unmix(Y, n_components=4):
    return ncpca_denoise_unmix(Y, n_components=3)


# ----------------------------------------------------------------------
# 6. Method 3: FLI-Net-style proxy (per-pixel temporal MLP regressor)
#
# IMPORTANT ARCHITECTURAL NOTE (corrected from an earlier version of this file):
# The published FLI-Net (Smith et al., PNAS 2019, "Fast fit-free analysis of
# fluorescence lifetime imaging via deep learning") is, despite being called a
# "3D CNN", explicitly SPATIALLY-INDEPENDENT: every convolution kernel in the
# architecture has spatial extent 1x1 (e.g. the first layer is a temporal-only
# Conv3D with kernel (1,1,10) and stride (1,1,5), followed by a (1,1,5) residual
# block, then 1x1 Conv2D/residual layers). The "3D" tensor operation is a
# vectorization trick that applies the same per-pixel temporal regression to
# every pixel of the cube at once -- it does NOT mix information across
# neighboring pixels. An earlier version of this proxy fed each network a 3x3
# spatial patch, which (unintentionally) gave the proxy MORE spatial context
# than real FLI-Net has. This version corrects that: each pixel's own decay
# curve (length T) is the only input, matching the real, spatially-independent
# design in spirit (a per-pixel nonlinear regressor from decay curve to
# abundance), implemented here as an MLP in place of the original Conv3D/
# residual-block/Keras-RMSprop training pipeline (no GPU deep-learning
# framework is available in this environment).
# ----------------------------------------------------------------------
from sklearn.neural_network import MLPRegressor


def extract_patches(Y, patch_radius=0):
    """patch_radius=0 reproduces FLI-Net's spatially-independent design:
    only the target pixel's own decay curve (length T) is used as input.
    A nonzero patch_radius is kept as an option for ablation but is not
    used for the FLI-Net-faithful benchmark."""
    H, W, T = Y.shape
    if patch_radius == 0:
        return Y.reshape(H * W, T)
    Yp = np.pad(Y, ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0)))
    feats = np.zeros((H * W, (2 * patch_radius + 1) ** 2 * T))
    idx = 0
    for i in range(H):
        for j in range(W):
            patch = Yp[i:i + 2 * patch_radius + 1, j:j + 2 * patch_radius + 1, :]
            feats[idx] = patch.reshape(-1)
            idx += 1
    return feats


def train_cnn_proxy(photons_per_bead, n_train_scenes=6, n_beads_train=15, patch_radius=0, seed=1):
    rng = np.random.default_rng(seed)
    X_list, Y_list = [], []
    for s in range(n_train_scenes):
        A_true, _ = generate_scene(n_beads_train, photons_per_bead, rng)
        Y_meas, _ = simulate_measurement(A_true, rng)
        feats = extract_patches(Y_meas, patch_radius)
        # Train against the PSF-blurred, noiseless target h*A_true (Eq.
        # "reblurred_component"), not the sharp source A_true: a
        # spatially-independent estimator (patch_radius=0) cannot recover
        # sharp, sub-PSF-resolution source locations from a single pixel's
        # own decay curve in principle, so the blurred target is the
        # correct, achievable regression objective -- see Section 2.6.
        target = np.stack([conv2d_same(A_true[k], PSF) for k in range(K_SPECIES)])
        labels = target.reshape(K_SPECIES, -1).T  # (H*W, K)
        X_list.append(feats)
        Y_list.append(labels)
    X_train = np.concatenate(X_list, axis=0)
    Y_train = np.concatenate(Y_list, axis=0)

    # Two hidden layers stand in for the shared temporal-feature branch
    # (Conv3D + residual block) of the original architecture; MLPRegressor's
    # default Adam solver is used in place of the paper's RMSprop(lr=1e-6)
    # since sklearn does not expose RMSprop for MLPRegressor.
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                          max_iter=400, random_state=seed, early_stopping=True)
    model.fit(X_train, Y_train)
    return model


def cnn_proxy_predict(model, Y, patch_radius=0):
    H, W, T = Y.shape
    feats = extract_patches(Y, patch_radius)
    pred = model.predict(feats)
    pred = np.clip(pred, 0, None)
    return pred.T.reshape(K_SPECIES, H, W)


# ----------------------------------------------------------------------
# 7. Method 4: Our method — joint spatial-temporal Poisson MLE
#    (multiplicative Richardson-Lucy/EM update with PSF coupling)
# ----------------------------------------------------------------------
def joint_spatiotemporal_mle(Y, n_iter=150, eta=0.9):
    H, W, T = Y.shape
    A = np.ones((K_SPECIES, H, W)) * (Y.sum() / (H * W * K_SPECIES) + EPS)
    C = np.stack([conv2d_same(np.ones((H, W)) * D_BASES[k].sum(), PSF) for k in range(K_SPECIES)])
    for _ in range(n_iter):
        blurred = np.stack([conv2d_same(A[k], PSF) for k in range(K_SPECIES)])  # (K,H,W)
        Lambda = np.einsum('khw,kt->hwt', blurred, D_BASES) + BACKGROUND_RATE
        R = Y / (Lambda + EPS)
        for k in range(K_SPECIES):
            temporal_proj = np.tensordot(R, D_BASES[k], axes=([2], [0]))  # (H,W)
            U_k = conv2d_same(temporal_proj, PSF)  # PSF symmetric -> adjoint = PSF
            A[k] *= (U_k / (C[k] + EPS)) ** eta
    return A


# ----------------------------------------------------------------------
# 8. Evaluation metrics
# ----------------------------------------------------------------------
def lifetime_at_points(A_est, coords, taus_ns, detection_floor=1e-3, fallback=None):
    """Amplitude-weighted mean lifetime at given (x,y) bead-center pixels.

    If the total predicted amplitude at a point is below `detection_floor`
    (i.e. the method predicted essentially no signal there -- a detection
    failure rather than a lifetime estimate), the lifetime is reported as
    `fallback` rather than as the raw weighted average, which would otherwise
    collapse toward 0 through division by EPS.

    `fallback` must be supplied explicitly whenever the reconstruction
    dictionary differs from the set of true lifetimes. Defaulting it to
    taus_ns.mean() ties the cost of a detection failure to the width of the
    dictionary: with the two true bases {2, 4} the default is 3.0 ns, and the
    agnostic 1.0-5.0 ns dictionary also has a mean of 3.0 ns. Supplying the
    fallback explicitly prevents later dictionary changes from silently changing
    the detection-failure penalty. That penalty is a property of the
    dictionary, not of the estimator, and must not enter the comparison.
    """
    taus = np.array(taus_ns)
    fb = taus.mean() if fallback is None else float(fallback)
    vals = []
    for (x, y) in coords:
        a = A_est[:, y, x]
        total = a.sum()
        vals.append(fb if total < detection_floor else (taus * a).sum() / total)
    return np.array(vals)

def evaluate_method(A_est, A_true, coords, species_true):
    tau_true = np.array(TAUS_NS)[species_true]
    tau_hat = lifetime_at_points(A_est, coords, TAUS_NS)
    lifetime_rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
    lifetime_bias = np.mean(tau_hat - tau_true)

    # intensity-map RMSE against noiseless expected detector-space image
    Lambda_true = forward_model(A_true).sum(axis=2)  # total expected counts per pixel
    intensity_est = A_est.sum(axis=0)  # source-space total abundance (for "our method")
    # For fair comparison, reblur to detector space when method output is source-space
    return lifetime_rmse, lifetime_bias, tau_hat, tau_true

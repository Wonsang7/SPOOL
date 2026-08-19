"""
Common FLIM low-photon simulation utilities for the SPOOL benchmarks.

Provides the 256-bin forward model used for two-lifetime (2 ns / 4 ns)
multi-emitter simulations, together with pixel-wise Poisson MLE, NC-PCA,
phasor analysis, sequential deconvolution controls, and SPOOL-related helpers.

Forward model
-------------
Lambda(r,t) = sum_k [h * A_k](r) * D_k(t) + B(r,t)
Y(r,t) ~ Poisson(Lambda(r,t))

Units: spatial in pixels, time in ns.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import nnls

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------
# 1. Physical / imaging parameters
# ----------------------------------------------------------------------
PIXEL_SIZE_NM = 50
PSF_FWHM_NM = 200.0
PSF_SIGMA_PX = (PSF_FWHM_NM / 2.3548200450309493) / PIXEL_SIZE_NM  # 1.699 px

IMAGE_SIZE = 100           # 100x100 px -> 5.0 um x 5.0 um FOV at 50 nm
N_TIME_BINS = 256
# Temporal axis matched to the experimental acquisitions and the
# single-emitter simulation. 256 bins x 0.0977 ns = 25.0 ns.
DT_NS = 0.0977
IRF_FWHM_NS = 0.150        # measured IRFs span 116-157 ps FWHM
IRF_PEAK_BIN = 5.1177      # t0 = 0.500 ns, as in the single-emitter study
TAUS_NS = [2, 4]           # two lifetime species
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
    simulations share one forward model.
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
    """Convert total expected photons per emitter to the peak-pixel count."""
    return np.asarray(photons_per_bead) * PSF_PEAK_HEIGHT


D_BASES = make_decay_bases(TAUS_NS, N_TIME_BINS, DT_NS, IRF_FWHM_NS)  # (K, T)
K_SPECIES = len(TAUS_NS)


def conv2d_same(img, kernel):
    return fftconvolve(img, kernel, mode="same")


# ----------------------------------------------------------------------
# 3. Ground-truth scene generation
# ----------------------------------------------------------------------
def generate_scene(n_beads, photons_per_bead, rng, margin=8, min_sep_px=1, n_species0=None):
    """Randomly place emitters on the pixel grid with a fixed balanced species count."""
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
# Pixel-wise Poisson MLE
# ----------------------------------------------------------------------
def pixelwise_poisson_mle(Y, n_iter=150, eta=1.0):
    H, W, T = Y.shape
    A = np.ones((K_SPECIES, H, W)) * (Y.sum() / (H * W * K_SPECIES) + EPS)
    for _ in range(n_iter):
        Lambda = np.einsum('khw,kt->hwt', A, D_BASES) + BACKGROUND_RATE
        R = Y / (Lambda + EPS)
        for k in range(K_SPECIES):
            U_k = np.tensordot(R, D_BASES[k], axes=([2], [0]))
            C_k = D_BASES[k].sum()
            A[k] *= (U_k / (C_k + EPS)) ** eta
    return A


# ----------------------------------------------------------------------
# Fit-free phasor analysis
# ----------------------------------------------------------------------
def phasor_lifetime_map(Y, dt_ns=DT_NS, freq_mhz=80.0):
    H, W, T = Y.shape
    t = np.arange(T) * dt_ns
    omega = 2 * np.pi * freq_mhz * 1e-3
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


# ----------------------------------------------------------------------
# Sequential unmixing/deconvolution utilities
# ----------------------------------------------------------------------
def pixelwise_nnls_unmix(Y):
    """Per-pixel temporal NNLS unmixing only, with no spatial processing."""
    H, W, T = Y.shape
    basis = D_BASES.T
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
    """Isotropic total-variation-regularized deconvolution."""
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
# Noise-corrected PCA followed by NNLS unmixing
# ----------------------------------------------------------------------
def ncpca_denoise_unmix(Y, n_components=3):
    H, W, T = Y.shape
    imgs = np.transpose(Y, (2, 0, 1))
    norm_factors = np.array([np.sqrt(max(img.mean(), EPS)) for img in imgs])
    imgs_norm = imgs / norm_factors[:, None, None]

    X = imgs_norm.reshape(T, H * W).T
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.decomposition import PCA
    r = min(n_components, T, H * W)
    pca = PCA(n_components=r, svd_solver="auto")
    X_pca = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_pca)

    recon_imgs_norm = X_recon.T.reshape(T, H, W)
    recon_imgs = recon_imgs_norm * norm_factors[:, None, None]
    recon_imgs = np.clip(recon_imgs, 0, None)
    flat_denoised = np.transpose(recon_imgs, (1, 2, 0)).reshape(H * W, T)

    basis = D_BASES.T
    A = np.zeros((K_SPECIES, H * W))
    for i in range(H * W):
        sol, _ = nnls(basis, flat_denoised[i])
        A[:, i] = sol
    return A.reshape(K_SPECIES, H, W)


def pca_denoise_unmix(Y, n_components=4):
    """Backward-compatible alias for the NC-PCA baseline."""
    return ncpca_denoise_unmix(Y, n_components=3)


# ----------------------------------------------------------------------
# Legacy supervised baseline helpers retained for reproducibility
# ----------------------------------------------------------------------
from sklearn.neural_network import MLPRegressor


def extract_patches(Y, patch_radius=0):
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
    for _ in range(n_train_scenes):
        A_true, _ = generate_scene(n_beads_train, photons_per_bead, rng)
        Y_meas, _ = simulate_measurement(A_true, rng)
        feats = extract_patches(Y_meas, patch_radius)
        target = np.stack([conv2d_same(A_true[k], PSF) for k in range(K_SPECIES)])
        labels = target.reshape(K_SPECIES, -1).T
        X_list.append(feats)
        Y_list.append(labels)
    X_train = np.concatenate(X_list, axis=0)
    Y_train = np.concatenate(Y_list, axis=0)
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
# Legacy NumPy SPOOL helper
# ----------------------------------------------------------------------
def joint_spatiotemporal_mle(Y, n_iter=150, eta=0.9):
    H, W, T = Y.shape
    A = np.ones((K_SPECIES, H, W)) * (Y.sum() / (H * W * K_SPECIES) + EPS)
    C = np.stack([conv2d_same(np.ones((H, W)) * D_BASES[k].sum(), PSF) for k in range(K_SPECIES)])
    for _ in range(n_iter):
        blurred = np.stack([conv2d_same(A[k], PSF) for k in range(K_SPECIES)])
        Lambda = np.einsum('khw,kt->hwt', blurred, D_BASES) + BACKGROUND_RATE
        R = Y / (Lambda + EPS)
        for k in range(K_SPECIES):
            temporal_proj = np.tensordot(R, D_BASES[k], axes=([2], [0]))
            U_k = conv2d_same(temporal_proj, PSF)
            A[k] *= (U_k / (C[k] + EPS)) ** eta
            A[k] = np.maximum(A[k], 1e-8)
    return A


# ----------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------
def lifetime_at_points(A_est, coords, taus_ns, detection_floor=1e-3, fallback=None):
    """Amplitude-weighted mean lifetime at given (x,y) emitter-center pixels."""
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
    return lifetime_rmse, lifetime_bias, tau_hat, tau_true

"""Runtime and GPU-memory benchmark for the manuscript dataset geometries.

The benchmark uses the manuscript-wide 50-iteration, K=11 reconstruction and
the same zero-padded linear convolution with a pixel-integrated Gaussian PSF.
Random Poisson counts and unit-area random dictionaries are used because the
computational cost is determined by array geometry, dtype, and iteration count.
"""

import os
import time

import numpy as np
import torch
from scipy.special import erf


N_ITER = 50
ETA = 0.9
K = 11
EPS = 1e-9
N_REP = 5
RUN_CPU = os.environ.get("SPOOL_RUNTIME_CPU", "1") != "0"

DATASETS = [
    ("beads",         100, 100, 256, 2.83),
    ("microtubules",  200, 200, 256, 2.18),
    ("cells",         520, 520, 150, 1.41),
    ("hyperspectral", 512, 512,  26, 1.70),
]


def make_ops(sig, height, width, dtype, device):
    """Build forward and adjoint OTFs from a pixel-integrated Gaussian PSF."""
    radius = int(np.ceil(4 * sig))
    axis = np.arange(-radius, radius + 1)
    weights = 0.5 * (
        erf((axis + 0.5) / (np.sqrt(2) * sig))
        - erf((axis - 0.5) / (np.sqrt(2) * sig))
    )
    psf = np.outer(weights, weights)
    psf /= psf.sum()

    ph, pw = psf.shape
    fh, fw = height + ph - 1, width + pw - 1
    pad = np.zeros((fh, fw))
    pad[:ph, :pw] = psf
    pad_flipped = np.zeros((fh, fw))
    pad_flipped[:ph, :pw] = psf[::-1, ::-1]
    otf = torch.fft.rfft2(
        torch.as_tensor(pad, dtype=dtype, device=device), s=(fh, fw)
    )
    adjoint_otf = torch.fft.rfft2(
        torch.as_tensor(pad_flipped, dtype=dtype, device=device), s=(fh, fw)
    )
    return {
        "otf": otf,
        "adj": adjoint_otf,
        "fs": (fh, fw),
        "out": (height, width),
        "off": ((ph - 1) // 2, (pw - 1) // 2),
    }


def conv(amplitudes, operators, adjoint=False):
    fh, fw = operators["fs"]
    height, width = operators["out"]
    y0, x0 = operators["off"]
    transfer = operators["adj"] if adjoint else operators["otf"]
    spectrum = torch.fft.rfft2(amplitudes, s=(fh, fw))
    full = torch.fft.irfft2(spectrum * transfer[None], s=(fh, fw))
    return full[:, y0:y0 + height, x0:x0 + width]


def joint(counts, dictionary, operators, normalizer, background, n_iter=N_ITER):
    n_components = dictionary.shape[0]
    height, width = counts.shape[:2]
    initial = max(float(counts.sum().item()) / (height * width * n_components), 1e-4)
    amplitudes = torch.full(
        (n_components, height, width), initial,
        dtype=counts.dtype, device=counts.device,
    )
    for _ in range(n_iter):
        blurred = conv(amplitudes, operators)
        expectation = torch.einsum("khw,kt->hwt", blurred, dictionary) + background
        ratio = counts / (expectation + EPS)
        update = conv(
            torch.einsum("hwt,kt->khw", ratio, dictionary),
            operators,
            adjoint=True,
        )
        amplitudes = torch.clamp(
            amplitudes * torch.clamp(update / (normalizer + EPS), min=0.0) ** ETA,
            min=1e-8,
        )
    return amplitudes


def bench(height, width, n_bins, sigma, dtype, device, synchronize):
    rng = np.random.default_rng(0)
    counts = torch.as_tensor(
        rng.poisson(2.0, (height, width, n_bins)).astype(np.float64),
        dtype=dtype,
        device=device,
    )
    raw_dictionary = np.abs(rng.normal(1, 0.3, (K, n_bins))) + 0.05
    dictionary = torch.as_tensor(
        raw_dictionary / raw_dictionary.sum(1, keepdims=True),
        dtype=dtype,
        device=device,
    )
    operators = make_ops(sigma, height, width, dtype, device)
    dictionary_sum = dictionary.sum(1).reshape(K, 1, 1)
    normalizer = conv(
        torch.ones((K, height, width), dtype=dtype, device=device), operators
    ) * dictionary_sum
    background = 0.01

    joint(counts, dictionary, operators, normalizer, background, n_iter=3)
    if synchronize:
        torch.cuda.synchronize()
    times = []
    for _ in range(N_REP):
        start = time.perf_counter()
        joint(counts, dictionary, operators, normalizer, background)
        if synchronize:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("The GPU runtime benchmark requires CUDA.")
    gpu = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}   iterations: {N_ITER}, K={K}")
    print(
        f"\n{'dataset':<14}{'shape':>16}{'GPU f64 (s)':>13}{'GPU f32 (s)':>13}"
        f"{'CPU f64 (s)':>13}{'peak GPU (GiB)':>16}"
    )
    for name, height, width, n_bins, sigma in DATASETS:
        torch.cuda.reset_peak_memory_stats()
        t64 = bench(height, width, n_bins, sigma, torch.float64, gpu, True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        t32 = bench(height, width, n_bins, sigma, torch.float32, gpu, True)
        tcpu = (
            bench(height, width, n_bins, sigma, torch.float64,
                  torch.device("cpu"), False)
            if RUN_CPU else float("nan")
        )
        shape = f"{height}x{width}x{n_bins}"
        print(
            f"{name:<14}{shape:>16}{t64:>13.3f}{t32:>13.3f}"
            f"{tcpu:>13.3f}{peak:>16.2f}"
        )

    print("\nTimes are medians of "
          f"{N_REP} runs at {N_ITER} iterations after a three-iteration warm-up.")
    print("Set SPOOL_RUNTIME_CPU=0 to skip the slower CPU timings.")


if __name__ == "__main__":
    main()

"""Adapter for the authors' official FPFLI implementation.

Reference
---------
D. Xiao et al., "Deep learning enhanced fast fluorescence lifetime imaging
with a few photons," Optica 10, 944-951 (2023).
https://doi.org/10.1364/OPTICA.491798

The adapter deliberately imports the official tauNet (LLE) and NIII classes
instead of reimplementing their networks. Input must have 256 temporal bins;
publication benchmarks should not upsample a coarse histogram to satisfy this
requirement because interpolation cannot restore missing timing information.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _torch_load(path: Path, device: torch.device):
    """Load old PyTorch checkpoints on both old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def gaussian_irf(
    n_bins: int = 256,
    dt_ns: float = 0.039,
    fwhm_ns: float = 0.1673,
    peak_bin: float = 15.0,
) -> np.ndarray:
    """Return the max-normalized IRF representation expected by FPFLI."""
    bins = np.arange(n_bins, dtype=np.float32)
    sigma_bins = fwhm_ns / (2.354820045 * dt_ns)
    irf = np.exp(-0.5 * ((bins - peak_bin) / sigma_bins) ** 2)
    return (irf / max(float(irf.max()), 1e-12))[None, :].astype(np.float32)


def _make_coord(shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    seqs = []
    for n in shape:
        r = 1.0 / n
        seqs.append(-1.0 + r + 2.0 * r * torch.arange(n, device=device))
    try:
        grid = torch.meshgrid(*seqs, indexing="ij")
    except TypeError:  # older PyTorch used by the original repository
        grid = torch.meshgrid(*seqs)
    return torch.stack(grid, dim=-1).reshape(-1, 2)


class OfficialFPFLI:
    """Run official FPFLI LLE and NIII weights on one FLIM decay cube."""

    def __init__(
        self,
        repo_root: str | Path,
        lle_weights: str | Path | None = None,
        niii_weights: str | Path | None = None,
        patch_size: int = 8,
        dt_ns: float = 0.039,
        pixel_threshold: float = 3.0,
        local_count_threshold: float = 50.0,
        device: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        eval_dir = self.repo_root / "Evaluation"
        if not (eval_dir / "tau_net.py").is_file():
            raise FileNotFoundError(
                f"Official FPFLI repository not found at {self.repo_root}. "
                "Clone https://github.com/Dooongx/FPFLI"
            )

        default_params = eval_dir / "model_parameters"
        self.lle_weights = Path(lle_weights) if lle_weights else (
            default_params / "LLE_parameter_1_3ns.pth"
        )
        self.niii_weights = Path(niii_weights) if niii_weights else (
            default_params / "NIII_parameter_L8.pth.tar"
        )
        for path in (self.lle_weights, self.niii_weights):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing official checkpoint: {path}\n"
                    "Download the checkpoint links listed in "
                    "Evaluation/model_parameters/readme.md in the FPFLI repository."
                )

        if patch_size != 8:
            raise ValueError("The supplied official NIII checkpoint is trained for patch_size=8")
        self.patch_size = patch_size
        self.dt_ns = float(dt_ns)
        self.pixel_threshold = float(pixel_threshold)
        self.local_count_threshold = float(local_count_threshold)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        tau_module = _load_module("fpfli_tau_net_official", eval_dir / "tau_net.py")
        niii_module = _load_module("fpfli_niii_official", eval_dir / "NIII_model.py")
        self.lle = tau_module.tauNet().to(self.device)
        self.niii = niii_module.NIII().to(self.device)

        lle_state = _torch_load(self.lle_weights, self.device)
        if isinstance(lle_state, dict) and "model" in lle_state:
            lle_state = lle_state["model"]
        self.lle.load_state_dict(lle_state)

        niii_state = _torch_load(self.niii_weights, self.device)
        if isinstance(niii_state, dict) and "model" in niii_state:
            niii_state = niii_state["model"]
        self.niii.load_state_dict(niii_state)
        self.lle.eval()
        self.niii.eval()

    @staticmethod
    def _pad_to_multiple(cube: np.ndarray, multiple: int):
        h, w, _ = cube.shape
        hp = ((h + multiple - 1) // multiple) * multiple
        wp = ((w + multiple - 1) // multiple) * multiple
        padded = np.pad(cube, ((0, hp - h), (0, wp - w), (0, 0)))
        return padded, (h, w)

    def _predict_local_lifetime(self, cube: np.ndarray, irf: np.ndarray) -> np.ndarray:
        p = self.patch_size
        hp, wp, n_bins = cube.shape
        blocks = cube.reshape(hp // p, p, wp // p, p, n_bins).sum(axis=(1, 3))
        flat = blocks.reshape(-1, n_bins).astype(np.float32)
        maxima = flat.max(axis=1, keepdims=True)
        flat = np.divide(flat, maxima, out=np.zeros_like(flat), where=maxima > 0)

        decay_t = torch.from_numpy(flat[:, None, :]).to(self.device)
        irf_t = torch.from_numpy(irf.astype(np.float32)).to(self.device)
        irf_t = irf_t.reshape(1, 1, n_bins).expand(decay_t.shape[0], -1, -1)
        with torch.inference_mode():
            tau = self.lle(decay_t, irf_t).detach().cpu().numpy()
        # This is the scaling used by the authors' Pred_local_lifetime.py.
        tau = tau * (self.dt_ns * 100.0)
        tau[blocks.sum(axis=2).reshape(-1) <= self.local_count_threshold] = 0.0
        return tau.reshape(hp // p, wp // p).astype(np.float32)

    def _run_niii(self, intensity: np.ndarray, local_tau: np.ndarray) -> np.ndarray:
        h, w = intensity.shape
        tau_max = max(float(local_tau.max()), 1e-8)
        int_max = max(float(intensity.max()), 1e-8)

        hr_int = torch.from_numpy((intensity / int_max).astype(np.float32))[None, None]
        lr_tau = torch.from_numpy((local_tau / tau_max).astype(np.float32))[None, None]
        hr_int = hr_int.to(self.device)
        lr_tau = lr_tau.to(self.device)
        lr_int = F.conv2d(
            hr_int,
            torch.ones((1, 1, self.patch_size, self.patch_size), device=self.device),
            stride=self.patch_size,
        )
        lr_up = F.interpolate(lr_tau, size=(h, w), mode="bicubic", align_corners=False)
        coord = _make_coord((h, w), self.device)[None]
        data = {
            "hr_int": hr_int,
            "lr_int": lr_int,
            "lr_tau": lr_tau,
            "lr_pixel": lr_up.reshape(1, -1, 1),
            "hr_coord": coord,
        }
        with torch.inference_mode():
            pred = self.niii(data) * tau_max
        return pred.reshape(h, w).detach().cpu().numpy().astype(np.float32)

    def predict(
        self,
        cube_hwt: np.ndarray,
        irf: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | str]:
        """Return full-resolution lifetime, local lifetime, mask, and intensity.

        `cube_hwt` is raw photon counts with shape (height, width, 256).
        The returned lifetime is detector-space; FPFLI does not invert a PSF.
        """
        cube = np.asarray(cube_hwt, dtype=np.float32)
        if cube.ndim != 3 or cube.shape[-1] != 256:
            raise ValueError(
                f"Official FPFLI requires (H,W,256), received {cube.shape}. "
                "Use a matched 256-bin simulator/acquisition; do not claim exact "
                "reproduction after naive temporal interpolation."
            )
        if np.any(cube < 0):
            raise ValueError("Photon counts must be non-negative")
        if irf is None:
            irf = gaussian_irf(dt_ns=self.dt_ns)
        irf = np.asarray(irf, dtype=np.float32).reshape(1, 256)
        irf /= max(float(irf.max()), 1e-12)

        padded, original_shape = self._pad_to_multiple(cube, self.patch_size)
        intensity_raw = padded.sum(axis=2)
        mask = intensity_raw > self.pixel_threshold
        masked = padded * mask[:, :, None]
        local_tau = self._predict_local_lifetime(masked, irf)
        tau = self._run_niii(masked.sum(axis=2), local_tau)
        tau[~mask] = 0.0

        h, w = original_shape
        return {
            "lifetime_ns": tau[:h, :w],
            "local_lifetime_ns": local_tau,
            "mask": mask[:h, :w],
            "intensity": intensity_raw[:h, :w],
            "space": "detector",
        }


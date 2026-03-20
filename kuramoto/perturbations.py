from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch

# -----------------------------
# Perturbations (plugin-style)
# -----------------------------
class Perturbation:
    """Interface: modifies dtheta for specific oscillators."""
    def apply(
        self,
        *,
        idx: int,
        t: float,
        theta: torch.Tensor,     # (B, N)
        dtheta: torch.Tensor,    # (B, N) base dynamics already computed
        omegas: torch.Tensor,    # (B, N)
        inter: torch.Tensor,     # (B, N)
    ) -> torch.Tensor:
        return dtheta


@dataclass
class GaussianPulse(Perturbation):
    t0: float
    tf: float
    source_idxs: list
    amp: float
    width: float

    def _pulse(self, t: float, device, dtype):
        tt = torch.tensor(t, device=device, dtype=dtype)
        return self.amp * torch.exp(-((tt - self.t0) ** 2) / (2.0 * (self.width ** 2)))

    def apply(self, *, t, theta, dtheta, omegas, inter, K):
        out = dtheta.clone()
        g = self._pulse(t, theta.device, theta.dtype)
        perturbation = torch.zeros_like(out)
        for idx in self.source_idxs:
                perturbation[:, idx] = g

        return perturbation


@dataclass
class CancelDynamics(Perturbation):
    t0: float
    tf: float
    source_idxs: list

    def apply(self, *, t, theta, dtheta, omegas, inter, K):
        out = dtheta.clone()
        perturbation = torch.zeros_like(out)
        for idx in self.source_idxs:
            perturbation[:, idx] = -1.0 * omegas[idx]  - (K / 4.0) * inter[:, idx] # (B,)
        return perturbation


@dataclass
class Disconnection(Perturbation): # Fixed spelling from Desconection
    t0: float
    tf: float
    source_idxs: list

    def apply(self, *, t, theta, dtheta, omegas, inter, K):
        out = dtheta.clone()
        perturbation = torch.zeros_like(out)
        for idx in self.source_idxs:
            perturbation[:, idx] =  - (K / 4.0) * inter[:, idx]
        return perturbation


@dataclass
class FreqDriftNoise(Perturbation):
    """
    Realistic perturbation: alters the local natural frequency (omega) with:
    - drift (slow bias)
    - Gaussian noise (jitter)
    """
    t0: float
    tf: float
    source_idxs: list
    drift: float          # constant change in dtheta (equivalent to delta_omega)
    sigma: float          # noise standard deviation


    def apply(self, *, t, theta, dtheta, omegas, inter, K):
        out = dtheta.clone()
        perturbation = torch.zeros_like(out)
        for idx in self.source_idxs:
            perturbation[:, idx] = self.drift + torch.randn(1, device=theta.device, dtype=theta.dtype) * self.sigma
        return perturbation


@dataclass
class DisconnectionTotal(Perturbation): # Fixed spelling from Desconection
    t0: float
    tf: float
    source_idxs: list

    def apply(self, *, t, theta, dtheta, omegas, inter, K, u_control):
        out = dtheta.clone()
        #print("u_control shape:", u_control.shape)
        perturbation = torch.zeros_like(out)
        for idx in self.source_idxs:
            perturbation[:, idx] =  - (K / 4.0) * inter[:, idx] #- u_control[:,idx]
        return perturbation
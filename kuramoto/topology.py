from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


@dataclass(frozen=True)
class Topology:
    A: np.ndarray
    L: np.ndarray
    lambda2: float
    lambda_max: float
    name: str
    params: Dict[str, Any]


def _require_nx():
    if nx is None:
        raise ImportError("networkx is required (pip install networkx).")


def _laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    D = np.diag(A.sum(axis=1))
    return D - A


def spectral_stats_from_laplacian(L: np.ndarray) -> Tuple[float, float]:
    eigvals = np.linalg.eigvalsh(L)
    eigvals.sort()
    lambda2 = float(eigvals[1]) if eigvals.shape[0] >= 2 else 0.0
    lambda_max = float(eigvals[-1]) if eigvals.shape[0] >= 1 else 0.0
    return lambda2, lambda_max


def build_graph(
    kind: str,
    *,
    n: Optional[int] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    periodic: bool = True,
    seed: Optional[int] = None,
    **params,
):
    """
    Returns (G, nodelist_or_None, name, used_params).

    kind in {"grid","ws","er"}:
      - grid: requires rows, cols. periodic=True gives torus
      - ws: requires n, k, p (seed optional)
      - er: requires n, p (seed optional)
    """
    _require_nx()
    kind = kind.lower().strip()

    if kind in {"grid", "grid2d"}:
        if rows is None or cols is None:
            raise ValueError("grid requires rows and cols.")
        G = nx.grid_2d_graph(int(rows), int(cols), periodic=bool(periodic))
        # consistent ordering for adjacency
        nodelist = [(r, c) for r in range(int(rows)) for c in range(int(cols))]
        name = f"grid2d({'torus' if periodic else 'open'})"
        used = {"rows": int(rows), "cols": int(cols), "periodic": bool(periodic)}

    elif kind in {"ws", "watts-strogatz", "watts_strogatz"}:
        if n is None:
            raise ValueError("ws requires n.")
        k = int(params.get("k"))
        p = float(params.get("p"))
        if k >= int(n):
            raise ValueError("WS requires k < n.")
        if k % 2 != 0:
            raise ValueError("WS requires k even.")
        G = nx.watts_strogatz_graph(n=int(n), k=k, p=p, seed=seed)
        nodelist = None
        name = "watts-strogatz"
        used = {"n": int(n), "k": k, "p": p, "seed": seed}

    elif kind in {"er", "erdos", "erdos-renyi", "erdos_renyi"}:
        if n is None:
            raise ValueError("er requires n.")
        p = float(params.get("p"))
        G = nx.erdos_renyi_graph(n=int(n), p=p, seed=seed)
        nodelist = None
        name = "erdos-renyi"
        used = {"n": int(n), "p": p, "seed": seed}

    else:
        raise ValueError(f"Unknown topology kind='{kind}'. Use: grid, ws, er.")

    return G, nodelist, name, used


def build_topology(
    kind: str,
    *,
    n: Optional[int] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    periodic: bool = True,
    seed: Optional[int] = None,
    **params,
) -> Topology:
    """
    Build Topology (A, L, lambda2) for kind in {"grid","ws","er"}.
    """
    G, nodelist, name, used_params = build_graph(
        kind, n=n, rows=rows, cols=cols, periodic=periodic, seed=seed, **params
    )

    A = nx.to_numpy_array(G, nodelist=nodelist, dtype=float)
    L = _laplacian_from_adjacency(A)
    lambda2, lambda_max = spectral_stats_from_laplacian(L)

    return Topology(
        A=A,
        L=L,
        lambda2=lambda2,
        lambda_max=lambda_max,
        name=name,
        params=used_params,
    )
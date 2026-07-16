"""
Matrix analysis utilities for connectivity matrices.

Pure functions over numpy arrays. No Connectome state, no dynamics,
no plasticity. Used after build_connectivity_matrix() to prepare C
for simulation:

    Connectome → Connectivity Matrix → Spectral Analysis → Spectral Normalization
"""

import numpy as np

DEFAULT_TARGET_SPECTRAL_RADIUS = 0.95


def spectral_radius(C: np.ndarray) -> float:
    """
    Return the spectral radius of C: max |λ_i| over eigenvalues λ_i.

    Determines whether linear activity x(t+1) = C @ x(t) tends to
    grow, decay, or persist.
    """
    eigenvalues = np.linalg.eigvals(C)
    return float(np.max(np.abs(eigenvalues)))


def normalize_spectral_radius(
    C: np.ndarray,
    target: float = DEFAULT_TARGET_SPECTRAL_RADIUS,
) -> np.ndarray:
    """
    Rescale C so its spectral radius equals target (default 0.95).

    Returns a new array. Callers that want a single matrix object
    should overwrite their reference:

        C = normalize_spectral_radius(C)

    The anatomical (unnormalized) matrix is not retained; regenerate
    it from Connectome via build_connectivity_matrix() if needed.
    """
    rho = spectral_radius(C)

    if rho == 0.0:
        return C.copy()

    return C * (target / rho)

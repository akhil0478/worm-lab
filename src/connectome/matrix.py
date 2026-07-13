"""
Build a numerical connectivity matrix from a Connectome.

Convention (frozen for v0.2):
    - Rows index destination neurons; columns index source neurons.
    - C[i, j] is the weight from neuron j to neuron i  (j → i).
    - Activity update: x(t+1) = C @ x(t),  x.shape == (n,).

The Connectome is the source of truth; this matrix is a derived view.
"""

import numpy as np

from src.connectome.models import Connectome


def build_connectivity_matrix(connectome: Connectome) -> np.ndarray:
    """
    Convert a Connectome into an (n, n) connectivity matrix.

    Chemical synapses (src → dst) contribute to C[dst_idx, src_idx].
    Gap junctions (A ↔ B) contribute symmetrically to C[i, j] and C[j, i].
    """
    n = len(connectome.neuron_list)
    C = np.zeros((n, n), dtype=np.float64)

    for (src, dst), weight in connectome.chemical_connections.items():
        src_idx = connectome.neuron_to_idx[src]
        dst_idx = connectome.neuron_to_idx[dst]
        C[dst_idx, src_idx] += weight

    for pair, weight in connectome.gap_junctions.items():
        nodes = list(pair)
        if len(nodes) != 2:
            continue
        a, b = nodes
        i = connectome.neuron_to_idx[a]
        j = connectome.neuron_to_idx[b]
        C[i, j] += weight
        C[j, i] += weight

    return C

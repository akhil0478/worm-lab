import numpy as np

from src.connectome.models import Connectome


def build_connectivity_matrix(connectome: Connectome) -> np.ndarray:
    """
    Convert a Connectome into an (n, n) connectivity matrix.

    Chemical synapses (src → dst) contribute to
    C[dst_idx, src_idx].

    Gap junctions (A ↔ B) contribute symmetrically
    to C[i, j] and C[j, i].
    """

    n = len(connectome.neuron_list)

    C = np.zeros(
        (n, n),
        dtype=np.float64
    )

    for (src, dst), weight in (
        connectome.chemical_connections.items()
    ):

        src_idx = connectome.neuron_to_idx[src]
        dst_idx = connectome.neuron_to_idx[dst]

        C[dst_idx, src_idx] += weight

    for pair, weight in (
        connectome.gap_junctions.items()
    ):

        nodes = list(pair)

        if len(nodes) != 2:
            continue

        a, b = nodes

        i = connectome.neuron_to_idx[a]
        j = connectome.neuron_to_idx[b]

        C[i, j] += weight
        C[j, i] += weight

    return C


def augment_with_nmj(
    C: np.ndarray,
    connectome: Connectome
) -> np.ndarray:
    """
    Add one simulated NMJ output node for each
    motor neuron with an NMJ connection.

    The original neurons occupy the first n indices.
    NMJ nodes occupy the remaining indices.

    NMJ nodes receive input from their corresponding
    motor neurons but have no outgoing connections.
    """

    n = len(connectome.neuron_list)
    nmj_count = len(connectome.nmj_connections)

    total = n + nmj_count

    C_aug = np.zeros(
        (total, total),
        dtype=np.float64
    )

    # ----------------------------------
    # COPY ORIGINAL CONNECTIVITY
    # ----------------------------------

    C_aug[:n, :n] = C

    # ----------------------------------
    # ADD NMJ CONNECTIONS
    # ----------------------------------

    for nmj_index, motor_neuron in enumerate(
        sorted(connectome.nmj_connections)
    ):

        motor_index = connectome.neuron_to_idx[
            motor_neuron
        ]

        output_index = n + nmj_index

        weight = connectome.nmj_connections[
            motor_neuron
        ]

        C_aug[
            output_index,
            motor_index
        ] = weight

    return C_aug
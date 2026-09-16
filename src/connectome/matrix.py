import numpy as np

from src.connectome.models import Connectome
from src.connectome.neuron_types import MOTOR_NEURONS


def build_connectivity_matrix(
    connectome: Connectome
) -> np.ndarray:

    n = len(connectome.neuron_list)

    C = np.zeros(
        (n, n),
        dtype=np.float64
    )

    # --------------------------------------------------
    # Chemical synapses
    # Convention:
    #
    # C[receiver, sender] = connection weight
    # --------------------------------------------------
    for (src, dst), weight in connectome.chemical_connections.items():

        src_idx = connectome.neuron_to_idx[src]
        dst_idx = connectome.neuron_to_idx[dst]

        C[dst_idx, src_idx] += weight

    # --------------------------------------------------
    # Gap junctions
    #
    # Gap junctions are bidirectional, so add:
    #
    # a -> b
    # b -> a
    # --------------------------------------------------
    for pair, weight in connectome.gap_junctions.items():

        nodes = list(pair)

        if len(nodes) != 2:
            continue

        a, b = nodes

        a_idx = connectome.neuron_to_idx[a]
        b_idx = connectome.neuron_to_idx[b]

        C[a_idx, b_idx] += weight
        C[b_idx, a_idx] += weight

    return C


def augment_with_nmj(
    C: np.ndarray,
    connectome: Connectome
) -> np.ndarray:

    # Number of biological neurons
    n = len(connectome.neuron_list)

    # Use the motor-neuron classification from neuron_types.py.
    #
    # The filtering ensures that we only use motor neurons
    # actually present in this particular connectome.
    motor_neurons = sorted(
        neuron
        for neuron in MOTOR_NEURONS
        if neuron in connectome.neuron_to_idx
    )

    # One artificial NMJ output neuron per biological motor neuron
    motor_count = len(motor_neurons)

    # Total number of neurons after augmentation
    total = n + motor_count

    # Create the expanded matrix
    C_aug = np.zeros(
        (total, total),
        dtype=np.float64
    )

    # Copy the original biological-to-biological connections
    #
    # Rows 0 through n-1:
    #     biological receiver neurons
    #
    # Columns 0 through n-1:
    #     biological sender neurons
    C_aug[:n, :n] = C

    # --------------------------------------------------
    # Add one artificial NMJ output neuron per motor neuron
    # --------------------------------------------------
    for nmj_index, motor_neuron in enumerate(motor_neurons):

        # Index of the biological motor neuron
        motor_index = connectome.neuron_to_idx[motor_neuron]

        # Artificial NMJ neurons are placed after
        # all biological neurons.
        output_index = n + nmj_index

        # Motor neuron sends activity to its artificial NMJ output.
        #
        # C_aug[receiver, sender] = weight
        #
        # Therefore:
        #     receiver = output_index
        #     sender   = motor_index
        C_aug[output_index, motor_index] = 1.0

    return C_aug
import numpy as np

from src.connectome.models import Connectome
from src.connectome.neuron_types import MOTOR_NEURONS
from src.reservoir.reservoir import Reservoir


def initialize_reservoir(
    connectome: Connectome,
    C_aug: np.ndarray,
    normal_threshold: float = 1.0,
    nmj_threshold: float = 2.0,
    retention: float = 0.97,
    reset: float = 0.01,
    learning_rate: float = 0.001,
) -> Reservoir:
    """
    Initialize the reservoir from the normalized augmented
    connectivity matrix.

    Biological neurons occupy the first n indices.
    Simulated NMJ neurons occupy the remaining indices.
    """

    biological_count = len(connectome.neuron_list)
    # Keep the size contract aligned with augment_with_nmj(): one simulated
    # NMJ node is appended for every classified motor neuron present in the
    # biological connectome, regardless of whether the source data has an
    # NMJ record for it.
    nmj_count = sum(
        neuron in connectome.neuron_to_idx
        for neuron in MOTOR_NEURONS
    )
    expected_size = biological_count + nmj_count

    if C_aug.shape != (expected_size, expected_size):
        raise ValueError(
            "Augmented matrix shape does not match connectome: "
            f"expected {(expected_size, expected_size)}, "
            f"got {C_aug.shape}"
        )

    size = C_aug.shape[0]

    # Anatomical baseline.
    C = C_aug.copy()

    # Mutable synaptic weights.
    # Learning will modify W, while C remains the baseline anatomy.
    W = C_aug.copy()

    # Membrane potentials and spike states.
    V = np.zeros(size, dtype=np.float64)
    v = np.zeros(size, dtype=np.float64)
    S = np.zeros(size, dtype=np.float64)

    # External input vector.
    I = np.zeros(size, dtype=np.float64)

    # Default thresholds.
    threshold = np.full(
        size,
        normal_threshold,
        dtype=np.float64,
    )

    # NMJ nodes are appended after the biological neurons.
    nmj_indices = np.arange(
        biological_count,
        size,
    )

    # NMJ nodes can use a separate threshold.
    threshold[nmj_indices] = nmj_threshold

    reservoir = Reservoir(
        C=C,
        V=V,
        W=W,
        I=I,
        S=S,
        v=v,
        threshold=threshold,
        reset=reset,
        learning_rate=learning_rate,
        retention=retention,
    )

    return reservoir

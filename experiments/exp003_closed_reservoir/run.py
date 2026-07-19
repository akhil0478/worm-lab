import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.connectome.loader import load_connectome
from src.connectome.matrix import build_connectivity_matrix
from src.connectome.metrics import normalize_spectral_radius

from src.reservoir.reservoir import Reservoir
from src.reservoir.activity import step


# ---------------------------------------
# Load connectome
# ---------------------------------------

connectome = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)


# ---------------------------------------
# Build normalized connectivity matrix
# ---------------------------------------

C = build_connectivity_matrix(connectome)

C = normalize_spectral_radius(C)


# ---------------------------------------
# Initial activity vector
# ---------------------------------------

V = np.zeros(len(connectome.neuron_list))

history = []

# Example stimulation

V[
    connectome.neuron_to_idx["AWCL"]
] = 1.0


# ---------------------------------------
# Create reservoir
# ---------------------------------------

reservoir = Reservoir(
    C=C,
    V=V
)


# ---------------------------------------
# Closed reservoir dynamics
# ---------------------------------------

print("=== CLOSED RESERVOIR (exp003) ===")
print("size:", reservoir.size)
print("stimulated neuron: AWCL")
print()

for t in range(50):

    activity_index, activity_vector = step(
        reservoir
    )

    print(
        f"Step {t+1:02d}"
    )

    print(
        "Activity Index:",
        activity_index
    )

    history.append(activity_vector)

    print("-" * 40)

history = np.array(history)
print("history shape")
print(history.shape)
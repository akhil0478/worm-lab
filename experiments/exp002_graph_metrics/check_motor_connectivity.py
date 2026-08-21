import sys
from pathlib import Path
import numpy as np
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root)) 

from src.connectome.loader import load_connectome
from src.connectome.matrix import build_connectivity_matrix


connectome = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

C = build_connectivity_matrix(connectome)

print("Motor neuron count:", len(connectome.motor_neurons))
print("Matrix shape:", C.shape)
print()

for name in sorted(connectome.motor_neurons):
    i = connectome.neuron_to_idx[name]

    incoming = np.count_nonzero(C[i, :])
    outgoing = np.count_nonzero(C[:, i])

    print(
        f"{name:>8} "
        f"index={i:3d} "
        f"incoming={incoming:2d} "
        f"outgoing={outgoing:2d}"
    )

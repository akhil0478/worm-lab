import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.connectome.loader import load_connectome

connectome = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

print("=== CONNECTOME SUMMARY ===")

print(
    "Total neurons:",
    len(connectome.neurons)
)

print(
    "Chemical connections:",
    len(connectome.chemical_connections)
)

print(
    "Gap junctions:",
    len(connectome.gap_junctions)
)

print(
    "Input neurons:",
    len(connectome.input_neurons)
)

print(
    "Motor neurons:",
    len(connectome.motor_neurons)
)

print(
    "NMJ output neurons:",
    len(connectome.output_neurons)
)

print("\nFirst 20 chemical connections:")

for i, ((src, dst), weight) in enumerate(
    connectome.chemical_connections.items()
):
    print(f"{src} -> {dst} : {weight}")

    if i >= 19:
        break

print("\nFirst 20 neuron indices:")

for neuron in connectome.neuron_list[:20]:
    print(
        f"{neuron} -> "
        f"{connectome.neuron_to_idx[neuron]}"
    )

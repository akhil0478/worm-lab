import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.connectome.loader import load_connectome

connectome = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

# ----------------------------------
# CHEMICAL GRAPH COVERAGE
# ----------------------------------

chemical_nodes = set()

for src, dst in connectome.chemical_connections.keys():
    chemical_nodes.add(src)
    chemical_nodes.add(dst)

missing = connectome.neurons - chemical_nodes

print("=" * 60)
print("CHEMICAL GRAPH COVERAGE")
print("=" * 60)

print("Total neurons:", len(connectome.neurons))
print("Chemical graph neurons:", len(chemical_nodes))
print("Missing neurons:", len(missing))

if missing:
    print("\nMissing neuron names:")
    print(sorted(missing))

# ----------------------------------
# ISOLATED NODES
# ----------------------------------

incoming = {n: 0 for n in connectome.neuron_list}
outgoing = {n: 0 for n in connectome.neuron_list}

for (src, dst), weight in connectome.chemical_connections.items():
    outgoing[src] += 1
    incoming[dst] += 1

isolated = []

for neuron in connectome.neuron_list:

    if (
        incoming[neuron] == 0
        and outgoing[neuron] == 0
    ):
        isolated.append(neuron)

print("\n" + "=" * 60)
print("ISOLATED NODES")
print("=" * 60)

print("Count:", len(isolated))

if isolated:
    print(sorted(isolated))

# ----------------------------------
# LOW DEGREE NODES
# ----------------------------------

print("\n" + "=" * 60)
print("LOW DEGREE NODES")
print("=" * 60)

for neuron in connectome.neuron_list:

    degree = (
        incoming[neuron]
        + outgoing[neuron]
    )

    if degree <= 2:
        print(
            f"{neuron:<8}"
            f" in={incoming[neuron]:<3}"
            f" out={outgoing[neuron]:<3}"
            f" total={degree}"
        )

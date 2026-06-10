import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.connectome.loader import load_connectome

connectome = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

weights = list(
    connectome.chemical_connections.values()
)

print("Chemical connections:",
      len(weights))

print("Min weight:",
      min(weights))

print("Max weight:",
      max(weights))

print("Average weight:",
      round(sum(weights) / len(weights), 2))

print("\nTop 20 strongest connections:")

top = sorted(
    connectome.chemical_connections.items(),
    key=lambda x: x[1],
    reverse=True
)

for (src, dst), weight in top[:20]:
    print(
        f"{src} -> {dst} : {weight}"
    )

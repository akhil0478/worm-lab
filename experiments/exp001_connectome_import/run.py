import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.connectome.loader import load_connectome

G = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


print("=== CONNECTOME SUMMARY ===")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("\nFirst 10 nodes:")
print(list(G.nodes())[:10])

print("\nFirst 10 edges:")
print(list(G.edges(data=True))[:10])

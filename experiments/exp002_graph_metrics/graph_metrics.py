import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.connectome.loader import load_connectome
import networkx as nx

G = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

avg_degree = (
    sum(dict(G.degree()).values())
    / G.number_of_nodes()
)
components = list(nx.weakly_connected_components(G))

print("Number of components:", len(components))
print("Largest component:", len(max(components, key=len)))

print("Average degree:", round(avg_degree, 2))

print("Weakly connected:",
      nx.is_weakly_connected(G))

print("Density:",
      nx.density(G))

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.connectome.loader import load_connectome


connectome = load_connectome(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

G = nx.DiGraph()

# Add neurons
for neuron in connectome.neurons:
    G.add_node(neuron)

# Add chemical connections
for (src, dst), weight in connectome.chemical_connections.items():
    G.add_edge(src, dst, weight=weight)

plt.figure(figsize=(16, 16))

pos = nx.spring_layout(
    G,
    k=0.3,
    iterations=50,
    seed=42
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=20,
    alpha=0.8
)

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.15,
    arrows=False
)

plt.title("C. elegans Connectome")
plt.axis("off")
plt.show()

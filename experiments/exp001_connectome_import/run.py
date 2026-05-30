from src.connectome.loader import load_connectome

G = load_connectome(
    "data/raw/connectome_dataset/connectome.csv"
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print(list(G.nodes()))
print(list(G.edges(data=True)))

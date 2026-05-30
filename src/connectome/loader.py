import pandas as pd
import networkx as nx


def load_connectome(filepath):
    df = pd.read_excel(filepath)

    # Remove neuromuscular junctions
    df = df[df["Type"] != "NMJ"]

    G = nx.DiGraph()

    for _, row in df.iterrows():
        source = row["Neuron 1"]
        target = row["Neuron 2"]
        weight = row["Nbr"]

        if G.has_edge(source, target):
            G[source][target]["weight"] += weight
        else:
            G.add_edge(
                source,
                target,
                weight=weight
            )

    return G

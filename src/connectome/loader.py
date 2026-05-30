import pandas as pd
import networkx as nx


def load_connectome(edge_file):
    """
    Load connectome edge list into a directed graph.
    """

    df = pd.read_csv(edge_file)

    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=row.get("weight", 1)
        )

    return G

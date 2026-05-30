from src.connectome.loader import load_connectome


def test_load_connectome():
    G = load_connectome(
        "data/raw/connectome_dataset/NeuronConnect1.xls"
    )

    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0

from src.connectome.loader import load_connectome


def test_load_connectome():
    connectome = load_connectome(
        "data/raw/connectome_dataset/NeuronConnect1.xls"
    )

    assert len(connectome.neurons) > 0
    assert len(connectome.chemical_connections) > 0
    assert len(connectome.input_neurons) > 0
    assert len(connectome.motor_neurons) > 0

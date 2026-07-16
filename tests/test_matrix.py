import numpy as np

from src.connectome.loader import load_connectome
from src.connectome.matrix import build_connectivity_matrix
from src.connectome.metrics import (
    DEFAULT_TARGET_SPECTRAL_RADIUS,
    normalize_spectral_radius,
    spectral_radius,
)


CONNECTOME_PATH = "data/raw/connectome_dataset/NeuronConnect1.xls"


def test_build_connectivity_matrix_shape_and_dtype():
    connectome = load_connectome(CONNECTOME_PATH)
    C = build_connectivity_matrix(connectome)

    n = len(connectome.neuron_list)
    assert C.shape == (n, n)
    assert C.dtype == np.float64
    assert np.count_nonzero(C) > 0


def test_chemical_encoding_destination_rows_source_columns():
    connectome = load_connectome(CONNECTOME_PATH)
    C = build_connectivity_matrix(connectome)

    (src, dst), weight = next(iter(connectome.chemical_connections.items()))
    src_idx = connectome.neuron_to_idx[src]
    dst_idx = connectome.neuron_to_idx[dst]

    assert C[dst_idx, src_idx] >= weight


def test_spectral_radius_raw_is_large():
    connectome = load_connectome(CONNECTOME_PATH)
    C = build_connectivity_matrix(connectome)

    rho = spectral_radius(C)
    assert rho > 1.0


def test_normalize_spectral_radius_targets_edge_of_stability():
    connectome = load_connectome(CONNECTOME_PATH)
    C = build_connectivity_matrix(connectome)

    C = normalize_spectral_radius(C)

    rho = spectral_radius(C)
    assert abs(rho - DEFAULT_TARGET_SPECTRAL_RADIUS) < 1e-6


def test_pipeline_overwrites_to_single_matrix():
    connectome = load_connectome(CONNECTOME_PATH)

    C = build_connectivity_matrix(connectome)
    C = normalize_spectral_radius(C)

    assert C.shape[0] == len(connectome.neuron_list)
    assert abs(spectral_radius(C) - 0.95) < 1e-6

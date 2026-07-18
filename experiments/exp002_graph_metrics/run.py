import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.connectome.loader import load_connectome
from src.connectome.matrix import build_connectivity_matrix
from src.connectome.metrics import spectral_radius, normalize_spectral_radius


def main() -> None:
    connectome = load_connectome(
        "data/raw/connectome_dataset/NeuronConnect1.xls"
    )

    C = build_connectivity_matrix(connectome)

    rho_before = spectral_radius(C)
    max_before = float(C.max())
    nnz_before = int(np.count_nonzero(C))

    C = normalize_spectral_radius(C)

    rho_after = spectral_radius(C)
    max_after = float(C.max())
    nnz_after = int(np.count_nonzero(C))

    n = len(connectome.neuron_list)

    print("=== MATRIX VERIFICATION (exp002) ===")
    print("n:", n)
    print("shape:", C.shape)
    print("dtype:", C.dtype)
    print()
    print("rho_before:", rho_before)
    print("rho_after :", rho_after)
    print("abs error :", abs(rho_after - 0.95))
    print()
    print("C.max() before:", max_before)
    print("C.max() after :", max_after)
    print()
    print("nonzero before:", nnz_before)
    print("nonzero after :", nnz_after)


if __name__ == "__main__":
    main()


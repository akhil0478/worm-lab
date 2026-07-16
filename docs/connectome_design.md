Dataset: NeuronConnect1.xls

Keep:
    S
    Sp
    EJ

Discard:
    R
    Rp

Chemical graph:
    (source,target) -> weight

Gap junctions:
    {a,b} -> weight

Inputs:
    34 sensory neurons

Outputs:
    63 motor neurons

Extended outputs:
    115 NMJ neurons

---

Matrix pipeline:

    Connectome
        → build_connectivity_matrix()   (matrix.py)
        → spectral_radius()             (metrics.py)
        → normalize_spectral_radius()   (metrics.py, target ρ = 0.95)

Rules:
    - Connectome is the sole anatomical source of truth.
    - Only one matrix object is kept: the spectrally normalized C.
    - The unnormalized matrix is not retained.
    - Regenerate anatomy via build_connectivity_matrix(connectome).
    - metrics.py holds pure functions on matrices (no wrapper class).
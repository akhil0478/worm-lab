import pandas as pd

from src.connectome.models import Connectome

from src.connectome.neuron_types import (
    SENSORY_NEURONS,
    MOTOR_NEURONS
)


def normalize_name(name):
    return str(name).strip().upper()


def load_connectome(filepath):
    df = pd.read_excel(filepath)

    # ----------------------------------
    # NORMALIZE NEURON NAMES
    # ----------------------------------

    df["Neuron 1"] = df["Neuron 1"].apply(
        normalize_name
    )

    df["Neuron 2"] = df["Neuron 2"].apply(
        normalize_name
    )

    # ----------------------------------
    # ALL NEURONS
    # ----------------------------------

    neurons = set(df["Neuron 1"]).union(
        set(df["Neuron 2"])
    )

    neuron_list = sorted(neurons)

    neuron_to_idx = {
        neuron: idx
        for idx, neuron in enumerate(neuron_list)
    }

    # ----------------------------------
    # OUTPUT NEURONS / NMJ CONNECTIONS
    # ----------------------------------

    nmj_df = df[
        df["Type"].astype(str).str.upper() == "NMJ"
    ]

    output_neurons = set(
        nmj_df["Neuron 1"]
    )

    motor_neurons = (
        output_neurons.intersection(
            MOTOR_NEURONS
        )
    )

    # Store only NMJ connections whose
    # source is an identified motor neuron.
    #
    # If a motor neuron has multiple NMJ
    # records, their weights are summed.
    #
    # Example:
    # DVB -> NMJ = 1
    # DVB -> NMJ = 4
    #
    # becomes:
    # DVB -> NMJ = 5

    nmj_connections = {}

    for _, row in nmj_df.iterrows():

        source = row["Neuron 1"]

        if source not in motor_neurons:
            continue

        weight = int(row["Nbr"])

        nmj_connections[source] = (
            nmj_connections.get(source, 0)
            + weight
        )

    # ----------------------------------
    # INPUT NEURONS
    # ----------------------------------

    input_neurons = (
        neurons.intersection(
            SENSORY_NEURONS
        )
    )

    # ----------------------------------
    # CHEMICAL CONNECTIONS
    # KEEP ONLY S + Sp
    # MERGE WEIGHTS
    # ----------------------------------

    chemical_connections = {}

    chem_df = df[
        df["Type"].astype(str).str.upper().isin(
            ["S", "SP"]
        )
    ]

    for _, row in chem_df.iterrows():

        source = row["Neuron 1"]
        target = row["Neuron 2"]

        weight = int(row["Nbr"])

        key = (source, target)

        chemical_connections[key] = (
            chemical_connections.get(key, 0)
            + weight
        )

    # ----------------------------------
    # GAP JUNCTIONS
    # ----------------------------------

    gap_junctions = {}

    ej_df = df[
        df["Type"].astype(str).str.upper() == "EJ"
    ]

    for _, row in ej_df.iterrows():

        a = row["Neuron 1"]
        b = row["Neuron 2"]

        weight = int(row["Nbr"])

        key = frozenset([a, b])

        gap_junctions[key] = (
            gap_junctions.get(key, 0)
            + weight
        )

    # ----------------------------------
    # CONNECTOME OBJECT
    # ----------------------------------

    return Connectome(
        neurons=neurons,

        neuron_list=neuron_list,
        neuron_to_idx=neuron_to_idx,

        chemical_connections=chemical_connections,
        gap_junctions=gap_junctions,
        nmj_connections=nmj_connections,

        input_neurons=input_neurons,
        motor_neurons=motor_neurons,
        output_neurons=output_neurons
    )
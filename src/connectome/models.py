from dataclasses import dataclass


@dataclass
class Connectome:
    neurons: set

    neuron_list: list
    neuron_to_idx: dict

    chemical_connections: dict
    gap_junctions: dict

    input_neurons: set
    motor_neurons: set
    output_neurons: set

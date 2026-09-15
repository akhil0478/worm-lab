from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from src.connectome.loader import load_connectome
from src.connectome.matrix import (
    build_connectivity_matrix,
    augment_with_nmj,
)
from src.connectome.metrics import normalize_spectral_radius

from src.reservoir.initialize import initialize_reservoir
from src.reservoir.activity import step


# ============================================================
# CONFIGURATION
# ============================================================

CONNECTOME_PATH = (
    ROOT
    / "data"
    / "raw"
    / "connectome_dataset"
    / "NeuronConnect1.xls"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "exp004_awcl_pulse_dynamics"
)

TIMESTEPS = 100
TARGET_RADIUS = 0.95

AWCL_NAME = "AWCL"

INPUT_STRENGTH = 1.0

SINGLE_PULSE_LENGTH = 1
FIVE_PULSE_LENGTH = 5


# ============================================================
# CONNECTOME / RESERVOIR CONSTRUCTION
# ============================================================

def build_reservoir():
    """
    Construct a fresh augmented and normalized reservoir.

    A new reservoir is created for every experiment so that
    the single-pulse and five-pulse experiments are independent.
    """

    connectome = load_connectome(CONNECTOME_PATH)

    # Build the biological connectome matrix.
    C_base = build_connectivity_matrix(connectome)

    # Add the aggregated NMJ/output nodes.
    #
    # The project function expects:
    #
    #     augment_with_nmj(C, connectome)
    #
    C_augmented = augment_with_nmj(
        C_base,
        connectome,
    )

    # Normalize the final matrix used by the reservoir.
    #
    # The project function uses "target", not "target_radius".
    C_normalized = normalize_spectral_radius(
        C_augmented,
        target=TARGET_RADIUS,
    )

    reservoir = initialize_reservoir(
        connectome=connectome,
        C_aug=C_normalized,
    )

    return connectome, reservoir


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_reservoir_size(reservoir):
    """
    Return the number of neurons in the reservoir.

    This avoids depending on a possibly missing Reservoir.size
    property.
    """

    return int(reservoir.V.shape[0])


def get_neuron_index(connectome, neuron_name):
    """
    Return the index of a biological neuron.

    The experiment uses neuron_to_idx because it defines the
    actual matrix indexing order.
    """

    try:
        return connectome.neuron_to_idx[neuron_name]

    except KeyError as exc:
        available = [
            name
            for name in connectome.neuron_list
            if neuron_name.lower() in name.lower()
        ]

        raise KeyError(
            f"Could not find neuron {neuron_name!r}. "
            f"Possible matches: {available}"
        ) from exc


def calculate_metrics(
    reservoir,
    connectome,
    awcl_index,
    first_spike_timestep,
    current_timestep,
):
    """
    Calculate network-level and neuron-level dynamics.
    """

    V = reservoir.V
    S = reservoir.S

    biological_count = len(connectome.neuron_list)

    biological_spikes = S[:biological_count]
    nmj_spikes = S[biological_count:]

    firing_indices = np.flatnonzero(S)

    # Record the first timestep on which each neuron fires.
    for index in firing_indices:
        if first_spike_timestep[index] == -1:
            first_spike_timestep[index] = current_timestep

    metrics = {
        "spikes": int(np.sum(S)),
        "active_neurons": int(np.count_nonzero(S)),
        "biological_spikes": int(np.sum(biological_spikes)),
        "nmj_spikes": int(np.sum(nmj_spikes)),
        "max_V": float(np.max(V)),
        "min_V": float(np.min(V)),
        "mean_V": float(np.mean(V)),
        "std_V": float(np.std(V)),
        "voltage_norm": float(np.linalg.norm(V)),
        "total_activity": float(np.sum(np.abs(V))),
        "awcl_V": float(V[awcl_index]),
        "awcl_spike": int(S[awcl_index]),
        "unique_neurons_fired": int(
            np.count_nonzero(first_spike_timestep != -1)
        ),
    }

    return metrics


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    pulse_length,
    label,
    timesteps=TIMESTEPS,
):
    """
    Run one experiment from a fresh reservoir.

    pulse_length=1:
        AWCL receives input only at t=0.

    pulse_length=5:
        AWCL receives input at t=0,1,2,3,4.
    """

    connectome, reservoir = build_reservoir()

    awcl_index = get_neuron_index(
        connectome,
        AWCL_NAME,
    )

    reservoir_size = get_reservoir_size(reservoir)
    biological_count = len(connectome.neuron_list)
    nmj_count = reservoir_size - biological_count

    # ========================================================
    # HISTORY ARRAYS
    # ========================================================

    spike_history = np.zeros(
        (timesteps, reservoir_size),
        dtype=np.float64,
    )

    voltage_history = np.zeros(
        (timesteps, reservoir_size),
        dtype=np.float64,
    )

    internal_voltage_history = np.zeros(
        (timesteps, reservoir_size),
        dtype=np.float64,
    )

    input_history = np.zeros(
        (timesteps, reservoir_size),
        dtype=np.float64,
    )

    metric_history = []

    # -1 means that the neuron has not fired yet.
    first_spike_timestep = np.full(
        reservoir_size,
        -1,
        dtype=int,
    )

    # ========================================================
    # EXPERIMENT INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print(f"EXPERIMENT: {label}")
    print("=" * 70)
    print(f"Reservoir size:      {reservoir_size}")
    print(f"Biological neurons:  {biological_count}")
    print(f"NMJ/output neurons:  {nmj_count}")
    print(f"AWCL index:          {awcl_index}")
    print(f"Pulse length:        {pulse_length}")
    print(f"Input strength:      {INPUT_STRENGTH}")
    print()

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for t in range(timesteps):

        # Clear all external input at every timestep.
        reservoir.I[:] = 0.0

        # Apply AWCL stimulation only during the pulse window.
        if t < pulse_length:
            reservoir.I[awcl_index] = INPUT_STRENGTH

        input_history[t] = reservoir.I.copy()

        # Advance the reservoir.
        V, internal_v = step(reservoir)

        # Record state.
        spike_history[t] = reservoir.S.copy()
        voltage_history[t] = V.copy()
        internal_voltage_history[t] = internal_v.copy()

        # Calculate metrics.
        metrics = calculate_metrics(
            reservoir=reservoir,
            connectome=connectome,
            awcl_index=awcl_index,
            first_spike_timestep=first_spike_timestep,
            current_timestep=t,
        )

        metrics["timestep"] = t
        metrics["input_active"] = int(t < pulse_length)

        metric_history.append(metrics)

        print(
            f"t={t + 1:03d} "
            f"input={int(t < pulse_length)} "
            f"spikes={metrics['spikes']:3d} "
            f"active={metrics['active_neurons']:3d} "
            f"bio={metrics['biological_spikes']:3d} "
            f"nmj={metrics['nmj_spikes']:3d} "
            f"unique={metrics['unique_neurons_fired']:3d} "
            f"max_V={metrics['max_V']:.5f} "
            f"mean_V={metrics['mean_V']:.5f} "
            f"AWCL_V={metrics['awcl_V']:.5f}"
        )

    # ========================================================
    # METRIC ARRAY
    # ========================================================

    metric_names = [
        "timestep",
        "input_active",
        "spikes",
        "active_neurons",
        "biological_spikes",
        "nmj_spikes",
        "unique_neurons_fired",
        "max_V",
        "min_V",
        "mean_V",
        "std_V",
        "voltage_norm",
        "total_activity",
        "awcl_V",
        "awcl_spike",
    ]

    metric_array = np.array(
        [
            [
                metrics[name]
                for metrics in metric_history
            ]
            for name in metric_names
        ],
        dtype=float,
    ).T

    # ========================================================
    # SAVE RAW EXPERIMENT DATA
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / f"{label}.npz"

    np.savez(
        output_path,
        spike_history=spike_history,
        voltage_history=voltage_history,
        internal_voltage_history=internal_voltage_history,
        input_history=input_history,
        metric_names=np.array(metric_names),
        metrics=metric_array,
        first_spike_timestep=first_spike_timestep,
        awcl_index=np.array(awcl_index),
        pulse_length=np.array(pulse_length),
    )

    print()
    print(f"Saved raw data to: {output_path}")

    return {
        "label": label,
        "connectome": connectome,
        "reservoir": reservoir,
        "spike_history": spike_history,
        "voltage_history": voltage_history,
        "internal_voltage_history": internal_voltage_history,
        "input_history": input_history,
        "metric_names": metric_names,
        "metrics": metric_array,
        "first_spike_timestep": first_spike_timestep,
        "awcl_index": awcl_index,
    }


# ============================================================
# PLOTTING
# ============================================================

def get_metric(results, metric_name):
    """
    Extract one metric from the saved metric matrix.
    """

    index = results["metric_names"].index(metric_name)

    return results["metrics"][:, index]


def plot_comparison(single, five):
    """
    Plot the most useful dynamics for both experiments.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    time_single = np.arange(
        len(single["spike_history"])
    )

    time_five = np.arange(
        len(five["spike_history"])
    )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13, 18),
        sharex=True,
    )

    # ========================================================
    # 1. NUMBER OF SPIKES
    # ========================================================

    axes[0].plot(
        time_single,
        get_metric(single, "spikes"),
        label="Single AWCL pulse",
    )

    axes[0].plot(
        time_five,
        get_metric(five, "spikes"),
        label="Five AWCL pulses",
    )

    axes[0].set_ylabel("Total spikes")
    axes[0].set_title("Spikes per timestep")
    axes[0].legend()
    axes[0].grid(True)

    # ========================================================
    # 2. UNIQUE NEURONS RECRUITED
    # ========================================================

    axes[1].plot(
        time_single,
        get_metric(single, "unique_neurons_fired"),
        label="Single AWCL pulse",
    )

    axes[1].plot(
        time_five,
        get_metric(five, "unique_neurons_fired"),
        label="Five AWCL pulses",
    )

    axes[1].set_ylabel("Unique neurons")
    axes[1].set_title("Cumulative neuron recruitment")
    axes[1].legend()
    axes[1].grid(True)

    # ========================================================
    # 3. MEAN AND MAXIMUM VOLTAGE
    # ========================================================

    axes[2].plot(
        time_single,
        get_metric(single, "max_V"),
        label="Single: max V",
    )

    axes[2].plot(
        time_single,
        get_metric(single, "mean_V"),
        label="Single: mean V",
    )

    axes[2].plot(
        time_five,
        get_metric(five, "max_V"),
        linestyle="--",
        label="Five: max V",
    )

    axes[2].plot(
        time_five,
        get_metric(five, "mean_V"),
        linestyle="--",
        label="Five: mean V",
    )

    axes[2].set_ylabel("Voltage")
    axes[2].set_title("Voltage dynamics")
    axes[2].legend()
    axes[2].grid(True)

    # ========================================================
    # 4. BIOLOGICAL VS NMJ SPIKES
    # ========================================================

    axes[3].plot(
        time_single,
        get_metric(single, "biological_spikes"),
        label="Single: biological",
    )

    axes[3].plot(
        time_single,
        get_metric(single, "nmj_spikes"),
        label="Single: NMJ",
    )

    axes[3].plot(
        time_five,
        get_metric(five, "biological_spikes"),
        linestyle="--",
        label="Five: biological",
    )

    axes[3].plot(
        time_five,
        get_metric(five, "nmj_spikes"),
        linestyle="--",
        label="Five: NMJ",
    )

    axes[3].set_ylabel("Spikes")
    axes[3].set_title("Biological and NMJ/output activity")
    axes[3].legend()
    axes[3].grid(True)

    # ========================================================
    # 5. AWCL VOLTAGE
    # ========================================================

    axes[4].plot(
        time_single,
        get_metric(single, "awcl_V"),
        label="Single AWCL pulse",
    )

    axes[4].plot(
        time_five,
        get_metric(five, "awcl_V"),
        label="Five AWCL pulses",
    )

    axes[4].axhline(
        1.0,
        linestyle=":",
        label="Threshold = 1.0",
    )

    axes[4].set_xlabel("Timestep")
    axes[4].set_ylabel("AWCL voltage")
    axes[4].set_title("AWCL membrane voltage")
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()

    plot_path = OUTPUT_DIR / "awcl_pulse_comparison.png"

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved comparison plot to: {plot_path}")

    plt.show()


# ============================================================
# SUMMARY
# ============================================================

def print_summary(single, five):
    """
    Print a compact comparison of the two experiments.
    """

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for result in [single, five]:

        label = result["label"]

        total_spikes = np.sum(
            get_metric(result, "spikes")
        )

        peak_spikes = np.max(
            get_metric(result, "spikes")
        )

        max_unique = np.max(
            get_metric(result, "unique_neurons_fired")
        )

        peak_voltage = np.max(
            get_metric(result, "max_V")
        )

        final_mean_voltage = get_metric(
            result,
            "mean_V",
        )[-1]

        total_biological_spikes = np.sum(
            get_metric(result, "biological_spikes")
        )

        total_nmj_spikes = np.sum(
            get_metric(result, "nmj_spikes")
        )

        print()
        print(label)
        print("-" * len(label))
        print(f"Total spikes:              {total_spikes:.0f}")
        print(f"Peak simultaneous spikes:  {peak_spikes:.0f}")
        print(f"Unique neurons fired:      {max_unique:.0f}")
        print(f"Peak voltage:              {peak_voltage:.5f}")
        print(f"Final mean voltage:        {final_mean_voltage:.5f}")
        print(
            "Biological spikes:         "
            f"{total_biological_spikes:.0f}"
        )
        print(
            "NMJ/output spikes:         "
            f"{total_nmj_spikes:.0f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    single_pulse = run_experiment(
        pulse_length=SINGLE_PULSE_LENGTH,
        label="awcl_single_pulse",
    )

    five_pulses = run_experiment(
        pulse_length=FIVE_PULSE_LENGTH,
        label="awcl_five_pulses",
    )

    print_summary(
        single_pulse,
        five_pulses,
    )

    plot_comparison(
        single_pulse,
        five_pulses,
    )
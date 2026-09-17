from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

# This file is designed to be executed from an interactive
# Python shell using:
#
#     exec(open("experiments/exp005_awcl_pulse_dynamics/run.py").read())
#
# Therefore, start Python from the project root:
#
#     cd /Users/akhilkrishna/worm-lab
#
ROOT = Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

import csv
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
    / "exp005_awcl_pulse_dynamics"
)

TIMESTEPS = 100
TARGET_RADIUS = 0.95

AWCL_NAME = "AWCL"

# The external current applied to AWCL.
INPUT_STRENGTH = 1.0

SINGLE_PULSE_LENGTH = 1
FIVE_PULSE_LENGTH = 5

# The important new part of Experiment 005:
# test several neuronal firing thresholds.
THRESHOLDS = [
    1.0,
    0.5,
    0.25,
    0.1,
    0.05,
    0.01,
]

# Set this to True if you want every timestep printed.
# False gives a much more useful summary.
PRINT_EVERY_TIMESTEP = False

# If True, print the first few active neurons for each run.
PRINT_FIRING_DETAILS = True

# Number of neuron names to print in diagnostic output.
MAX_NEURONS_TO_PRINT = 25


# ============================================================
# PROJECT / RESERVOIR CONSTRUCTION
# ============================================================

def build_reservoir(threshold):
    """
    Construct a fresh augmented and normalized reservoir.

    A new reservoir is created for every threshold and pulse
    experiment so that experiments remain independent.
    """

    connectome = load_connectome(CONNECTOME_PATH)

    # Build biological connectivity matrix.
    C_base = build_connectivity_matrix(connectome)

    # Add artificial NMJ/output neurons.
    C_augmented = augment_with_nmj(
        C_base,
        connectome,
    )

    # Normalize the final matrix.
    C_normalized = normalize_spectral_radius(
        C_augmented,
        target=TARGET_RADIUS,
    )

    reservoir = initialize_reservoir(
        connectome=connectome,
        C_aug=C_normalized,
    )

    # Set the firing threshold for this experiment.
    #
    # The initialization function creates the threshold array.
    # We replace all neuron thresholds with the selected scalar.
    if np.isscalar(reservoir.threshold):
        reservoir.threshold = float(threshold)
    else:
        reservoir.threshold[:] = float(threshold)

    return connectome, reservoir


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_reservoir_size(reservoir):
    """
    Return the number of neurons in the reservoir.
    """

    return int(reservoir.V.shape[0])


def get_neuron_index(connectome, neuron_name):
    """
    Return the matrix index of a biological neuron.
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


def get_threshold_value(reservoir):
    """
    Return the threshold as a scalar for printing.
    """

    if np.isscalar(reservoir.threshold):
        return float(reservoir.threshold)

    return float(np.max(reservoir.threshold))


def get_neuron_names(connectome, indices):
    """
    Convert neuron indices into neuron names where possible.

    NMJ/output neurons are artificial nodes and may not exist
    in connectome.neuron_list, so they receive generated names.
    """

    names = []

    biological_count = len(connectome.neuron_list)

    for index in indices:
        index = int(index)

        if index < biological_count:
            names.append(connectome.neuron_list[index])
        else:
            names.append(f"NMJ_OUTPUT_{index - biological_count}")

    return names


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

    # Record first timestep on which each neuron fires.
    for index in firing_indices:
        if first_spike_timestep[index] == -1:
            first_spike_timestep[index] = current_timestep

    metrics = {
        "timestep": current_timestep,
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


def print_firing_details(
    result,
    connectome,
    threshold,
    pulse_length,
):
    """
    Print useful neuron-level information for one experiment.
    """

    spike_history = result["spike_history"]
    first_spike_timestep = result["first_spike_timestep"]

    ever_fired = np.flatnonzero(first_spike_timestep != -1)

    downstream_indices = [
        int(index)
        for index in ever_fired
        if int(index) != result["awcl_index"]
    ]

    print()
    print(
        f"Threshold={threshold:.3f}, "
        f"pulse_length={pulse_length}"
    )
    print(
        f"Total neurons that fired: {len(ever_fired)}"
    )
    print(
        f"Downstream neurons excluding AWCL: "
        f"{len(downstream_indices)}"
    )

    if len(downstream_indices) == 0:
        print("No downstream neurons fired.")

    else:
        names = get_neuron_names(
            connectome,
            downstream_indices[:MAX_NEURONS_TO_PRINT],
        )

        print("First downstream neurons fired:")

        for index, name in zip(
            downstream_indices[:MAX_NEURONS_TO_PRINT],
            names,
        ):
            print(
                f"  index={index:3d} "
                f"name={name:20s} "
                f"first_t={first_spike_timestep[index]}"
            )

        if len(downstream_indices) > MAX_NEURONS_TO_PRINT:
            print(
                f"  ... and "
                f"{len(downstream_indices) - MAX_NEURONS_TO_PRINT} "
                f"more neurons"
            )

    # Report which timesteps contained activity after the input.
    total_spikes_per_timestep = np.sum(
        spike_history,
        axis=1,
    )

    downstream_spikes_per_timestep = np.sum(
        spike_history[:, downstream_indices],
        axis=1,
    ) if downstream_indices else np.zeros(
        spike_history.shape[0]
    )

    downstream_timesteps = np.flatnonzero(
        downstream_spikes_per_timestep > 0
    )

    if len(downstream_timesteps) > 0:
        print(
            "First downstream spike timestep: "
            f"{int(downstream_timesteps[0])}"
        )
        print(
            "Last downstream spike timestep: "
            f"{int(downstream_timesteps[-1])}"
        )
    else:
        print("No downstream spike timesteps detected.")

    print(
        "Peak total simultaneous spikes: "
        f"{int(np.max(total_spikes_per_timestep))}"
    )


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    threshold,
    pulse_length,
    label,
    timesteps=TIMESTEPS,
):
    """
    Run one experiment using one threshold and one pulse length.
    """

    connectome, reservoir = build_reservoir(
        threshold=threshold,
    )

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
    print(f"Threshold:           {threshold}")
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

        # Clear external input.
        reservoir.I[:] = 0.0

        # Apply AWCL stimulation during the pulse window.
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

        metrics["input_active"] = int(t < pulse_length)

        metric_history.append(metrics)

        if PRINT_EVERY_TIMESTEP:
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
    # RESULT OBJECT
    # ========================================================

    result = {
        "label": label,
        "threshold": threshold,
        "pulse_length": pulse_length,
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

    # ========================================================
    # SAVE RAW DATA
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
        threshold=np.array(threshold),
    )

    print(f"Saved raw data to: {output_path}")

    if PRINT_FIRING_DETAILS:
        print_firing_details(
            result=result,
            connectome=connectome,
            threshold=threshold,
            pulse_length=pulse_length,
        )

    return result


# ============================================================
# METRIC ACCESS
# ============================================================

def get_metric(results, metric_name):
    """
    Extract one metric from the metric matrix.
    """

    index = results["metric_names"].index(metric_name)

    return results["metrics"][:, index]


def summarize_result(result):
    """
    Convert one experiment result into a compact dictionary.
    """

    spike_counts = get_metric(
        result,
        "spikes",
    )

    unique_counts = get_metric(
        result,
        "unique_neurons_fired",
    )

    biological_spikes = get_metric(
        result,
        "biological_spikes",
    )

    nmj_spikes = get_metric(
        result,
        "nmj_spikes",
    )

    awcl_voltage = get_metric(
        result,
        "awcl_V",
    )

    # Determine whether anything other than AWCL fired.
    first_spike_timestep = result["first_spike_timestep"]

    ever_fired = np.flatnonzero(
        first_spike_timestep != -1
    )

    downstream_fired = [
        int(index)
        for index in ever_fired
        if int(index) != result["awcl_index"]
    ]

    downstream_spike_count = int(
        np.sum(
            result["spike_history"][
                :,
                downstream_fired,
            ]
        )
    ) if downstream_fired else 0

    return {
        "threshold": result["threshold"],
        "pulse_length": result["pulse_length"],
        "total_spikes": int(np.sum(spike_counts)),
        "peak_simultaneous_spikes": int(np.max(spike_counts)),
        "unique_neurons_fired": int(np.max(unique_counts)),
        "downstream_neurons_fired": int(
            len(downstream_fired)
        ),
        "downstream_spikes": downstream_spike_count,
        "biological_spikes": int(
            np.sum(biological_spikes)
        ),
        "nmj_spikes": int(
            np.sum(nmj_spikes)
        ),
        "peak_voltage": float(
            np.max(
                get_metric(
                    result,
                    "max_V",
                )
            )
        ),
        "peak_awcl_voltage": float(
            np.max(awcl_voltage)
        ),
        "final_mean_voltage": float(
            get_metric(
                result,
                "mean_V",
            )[-1]
        ),
        "first_downstream_timestep": (
            int(
                np.min(
                    first_spike_timestep[
                        downstream_fired
                    ]
                )
            )
            if downstream_fired
            else -1
        ),
    }


# ============================================================
# SUMMARY PRINTING
# ============================================================

def print_threshold_summary(summary_rows):
    """
    Print a compact comparison across all thresholds.
    """

    print()
    print("=" * 120)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 120)

    headers = [
        "threshold",
        "pulse",
        "total",
        "peak",
        "unique",
        "downstream",
        "down_spikes",
        "bio",
        "nmj",
        "peak_V",
        "AWCL_peak",
        "first_downstream",
    ]

    print(
        f"{headers[0]:>10} "
        f"{headers[1]:>6} "
        f"{headers[2]:>8} "
        f"{headers[3]:>8} "
        f"{headers[4]:>8} "
        f"{headers[5]:>11} "
        f"{headers[6]:>11} "
        f"{headers[7]:>8} "
        f"{headers[8]:>8} "
        f"{headers[9]:>10} "
        f"{headers[10]:>11} "
        f"{headers[11]:>16}"
    )

    print("-" * 120)

    for row in summary_rows:
        first_downstream = row["first_downstream_timestep"]

        first_downstream_text = (
            str(first_downstream)
            if first_downstream >= 0
            else "none"
        )

        print(
            f"{row['threshold']:10.3f} "
            f"{row['pulse_length']:6d} "
            f"{row['total_spikes']:8d} "
            f"{row['peak_simultaneous_spikes']:8d} "
            f"{row['unique_neurons_fired']:8d} "
            f"{row['downstream_neurons_fired']:11d} "
            f"{row['downstream_spikes']:11d} "
            f"{row['biological_spikes']:8d} "
            f"{row['nmj_spikes']:8d} "
            f"{row['peak_voltage']:10.5f} "
            f"{row['peak_awcl_voltage']:11.5f} "
            f"{first_downstream_text:>16}"
        )


def save_summary_csv(summary_rows):
    """
    Save the threshold sweep summary as a CSV file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = OUTPUT_DIR / "threshold_sweep_summary.csv"

    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Saved summary CSV to: {csv_path}")


# ============================================================
# PLOTTING
# ============================================================

def plot_threshold_comparison(
    results_by_threshold,
    pulse_length,
):
    """
    Plot how activity changes with firing threshold.
    """

    thresholds = sorted(
        results_by_threshold.keys(),
        reverse=True,
    )

    summaries = [
        summarize_result(
            results_by_threshold[threshold]
        )
        for threshold in thresholds
    ]

    threshold_labels = [
        str(threshold)
        for threshold in thresholds
    ]

    total_spikes = [
        row["total_spikes"]
        for row in summaries
    ]

    unique_neurons = [
        row["unique_neurons_fired"]
        for row in summaries
    ]

    downstream_neurons = [
        row["downstream_neurons_fired"]
        for row in summaries
    ]

    downstream_spikes = [
        row["downstream_spikes"]
        for row in summaries
    ]

    peak_voltages = [
        row["peak_voltage"]
        for row in summaries
    ]

    nmj_spikes = [
        row["nmj_spikes"]
        for row in summaries
    ]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 14),
    )

    # --------------------------------------------------------
    # Total spikes
    # --------------------------------------------------------

    axes[0, 0].plot(
        threshold_labels,
        total_spikes,
        marker="o",
    )

    axes[0, 0].set_title(
        f"Total spikes vs threshold "
        f"(pulse length={pulse_length})"
    )
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Total spikes")
    axes[0, 0].grid(True)

    # --------------------------------------------------------
    # Unique neurons
    # --------------------------------------------------------

    axes[0, 1].plot(
        threshold_labels,
        unique_neurons,
        marker="o",
    )

    axes[0, 1].set_title("Unique neurons recruited")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Unique neurons")
    axes[0, 1].grid(True)

    # --------------------------------------------------------
    # Downstream neurons
    # --------------------------------------------------------

    axes[1, 0].plot(
        threshold_labels,
        downstream_neurons,
        marker="o",
    )

    axes[1, 0].set_title(
        "Downstream neurons excluding AWCL"
    )
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Downstream neurons")
    axes[1, 0].grid(True)

    # --------------------------------------------------------
    # Downstream spikes
    # --------------------------------------------------------

    axes[1, 1].plot(
        threshold_labels,
        downstream_spikes,
        marker="o",
    )

    axes[1, 1].set_title("Downstream spike count")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Downstream spikes")
    axes[1, 1].grid(True)

    # --------------------------------------------------------
    # Peak voltage
    # --------------------------------------------------------

    axes[2, 0].plot(
        threshold_labels,
        peak_voltages,
        marker="o",
    )

    axes[2, 0].set_title("Peak network voltage")
    axes[2, 0].set_xlabel("Threshold")
    axes[2, 0].set_ylabel("Peak voltage")
    axes[2, 0].grid(True)

    # --------------------------------------------------------
    # NMJ spikes
    # --------------------------------------------------------

    axes[2, 1].plot(
        threshold_labels,
        nmj_spikes,
        marker="o",
    )

    axes[2, 1].set_title("NMJ/output spikes")
    axes[2, 1].set_xlabel("Threshold")
    axes[2, 1].set_ylabel("NMJ spikes")
    axes[2, 1].grid(True)

    plt.suptitle(
        f"AWCL threshold sweep "
        f"(pulse length={pulse_length})",
        fontsize=16,
    )

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / f"threshold_comparison_pulse_{pulse_length}.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved threshold comparison plot to: "
        f"{plot_path}"
    )

    plt.show()


def plot_activity_heatmap(
    results_by_threshold,
    pulse_length,
):
    """
    Plot a heatmap showing which neurons fired over time.

    Each row corresponds to one threshold.
    Each column corresponds to a timestep.
    Values represent total spikes at that timestep.
    """

    thresholds = sorted(
        results_by_threshold.keys(),
        reverse=True,
    )

    activity_matrix = []

    for threshold in thresholds:
        result = results_by_threshold[threshold]

        spikes_per_timestep = np.sum(
            result["spike_history"],
            axis=1,
        )

        activity_matrix.append(
            spikes_per_timestep
        )

    activity_matrix = np.array(
        activity_matrix,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(15, 6),
    )

    image = ax.imshow(
        activity_matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(
        f"Total spiking activity across thresholds "
        f"(pulse length={pulse_length})"
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Threshold")

    ax.set_yticks(
        np.arange(len(thresholds))
    )
    ax.set_yticklabels(
        [
            str(threshold)
            for threshold in thresholds
        ]
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of spikes",
    )

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / f"activity_heatmap_pulse_{pulse_length}.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved activity heatmap to: "
        f"{plot_path}"
    )

    plt.show()


def plot_neuron_heatmap(
    result,
):
    """
    Plot the neuron-by-time spike matrix for one result.
    """

    spike_history = result["spike_history"]

    fig, ax = plt.subplots(
        figsize=(15, 8),
    )

    image = ax.imshow(
        spike_history.T,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(
        f"Neuron activity: "
        f"threshold={result['threshold']}, "
        f"pulse={result['pulse_length']}"
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron index")

    fig.colorbar(
        image,
        ax=ax,
        label="Spike state",
    )

    plt.tight_layout()

    threshold_text = str(
        result["threshold"]
    ).replace(".", "p")

    plot_path = (
        OUTPUT_DIR
        / (
            f"neuron_heatmap_threshold_"
            f"{threshold_text}_pulse_"
            f"{result['pulse_length']}.png"
        )
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved neuron heatmap to: "
        f"{plot_path}"
    )

    plt.show()


# ============================================================
# FULL THRESHOLD SWEEP
# ============================================================

def run_threshold_sweep():
    """
    Run both pulse experiments for every threshold.
    """

    all_results = {}
    summary_rows = []

    for threshold in THRESHOLDS:

        print()
        print("#" * 90)
        print(
            f"STARTING THRESHOLD SWEEP VALUE: "
            f"{threshold}"
        )
        print("#" * 90)

        threshold_results = {}

        # ----------------------------------------------------
        # Single pulse
        # ----------------------------------------------------

        single_label = (
            f"awcl_single_pulse_threshold_"
            f"{str(threshold).replace('.', 'p')}"
        )

        single_result = run_experiment(
            threshold=threshold,
            pulse_length=SINGLE_PULSE_LENGTH,
            label=single_label,
        )

        threshold_results["single"] = single_result

        summary_rows.append(
            summarize_result(single_result)
        )

        # ----------------------------------------------------
        # Five pulses
        # ----------------------------------------------------

        five_label = (
            f"awcl_five_pulses_threshold_"
            f"{str(threshold).replace('.', 'p')}"
        )

        five_result = run_experiment(
            threshold=threshold,
            pulse_length=FIVE_PULSE_LENGTH,
            label=five_label,
        )

        threshold_results["five"] = five_result

        summary_rows.append(
            summarize_result(five_result)
        )

        all_results[threshold] = threshold_results

    return all_results, summary_rows


# ============================================================
# FINAL INTERPRETATION
# ============================================================

def print_interpretation(summary_rows):
    """
    Print an automatically generated interpretation of the
    threshold sweep.
    """

    print()
    print("=" * 90)
    print("INTERPRETATION OF THRESHOLD SWEEP")
    print("=" * 90)

    for pulse_length in [
        SINGLE_PULSE_LENGTH,
        FIVE_PULSE_LENGTH,
    ]:

        rows = [
            row
            for row in summary_rows
            if row["pulse_length"] == pulse_length
        ]

        rows = sorted(
            rows,
            key=lambda row: row["threshold"],
            reverse=True,
        )

        print()
        print(
            f"Pulse length = {pulse_length}"
        )
        print("-" * 90)

        for row in rows:
            threshold = row["threshold"]

            if row["downstream_neurons_fired"] > 0:
                status = (
                    "DOWNSTREAM ACTIVITY DETECTED"
                )

            elif row["total_spikes"] > pulse_length:
                status = (
                    "REPEATED OR ADDITIONAL ACTIVITY, "
                    "BUT NO DOWNSTREAM NEURONS"
                )

            else:
                status = (
                    "AWCL-ONLY RESPONSE"
                )

            print(
                f"threshold={threshold:>6.3f} | "
                f"total_spikes={row['total_spikes']:>5d} | "
                f"unique={row['unique_neurons_fired']:>4d} | "
                f"downstream={row['downstream_neurons_fired']:>4d} | "
                f"NMJ={row['nmj_spikes']:>4d} | "
                f"{status}"
            )

    print()
    print(
        "Interpretation notes:"
    )
    print(
        "1. Lowering the threshold should make neurons easier "
        "to activate, but it cannot create connectivity that "
        "does not exist."
    )
    print(
        "2. If AWCL remains the only firing neuron at every "
        "threshold, inspect AWCL's outgoing connections and "
        "the sign/magnitude of those weights."
    )
    print(
        "3. If many neurons fire at threshold=0.01, the network "
        "may be entering a highly excitable or runaway regime."
    )
    print(
        "4. NMJ spikes indicate that artificial output neurons "
        "are receiving enough activation to cross threshold."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Execute the complete threshold sweep.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results, summary_rows = run_threshold_sweep()

    print_threshold_summary(
        summary_rows
    )

    save_summary_csv(
        summary_rows
    )

    # Generate plots for single-pulse experiments.
    single_results = {
        threshold: results["single"]
        for threshold, results in all_results.items()
    }

    plot_threshold_comparison(
        results_by_threshold=single_results,
        pulse_length=SINGLE_PULSE_LENGTH,
    )

    plot_activity_heatmap(
        results_by_threshold=single_results,
        pulse_length=SINGLE_PULSE_LENGTH,
    )

    # Generate plots for five-pulse experiments.
    five_results = {
        threshold: results["five"]
        for threshold, results in all_results.items()
    }

    plot_threshold_comparison(
        results_by_threshold=five_results,
        pulse_length=FIVE_PULSE_LENGTH,
    )

    plot_activity_heatmap(
        results_by_threshold=five_results,
        pulse_length=FIVE_PULSE_LENGTH,
    )

    # Plot detailed neuron heatmaps only for the lowest
    # threshold, where activity is most likely to propagate.
    lowest_threshold = min(THRESHOLDS)

    plot_neuron_heatmap(
        single_results[lowest_threshold]
    )

    plot_neuron_heatmap(
        five_results[lowest_threshold]
    )

    print_interpretation(
        summary_rows
    )

    print()
    print("=" * 90)
    print("EXPERIMENT 005 COMPLETE")
    print("=" * 90)
    print(
        f"Thresholds tested: {THRESHOLDS}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


# ============================================================
# INTERACTIVE EXECUTION SUPPORT
# ============================================================

# This condition is true when the file is executed through:
#
#     exec(open("experiments/exp005_awcl_pulse_dynamics/run.py").read())
#
# because __name__ will normally be "__main__" in the
# interactive interpreter.
#
# It is also safe to call main() manually after loading the file.

if __name__ == "__main__":
    main()

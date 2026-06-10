import pandas as pd

file_path = "data/raw/connectome_dataset/NeuronConnect1.xls"

xls = pd.ExcelFile(file_path)

print("Sheets:")
print(xls.sheet_names)

for sheet in xls.sheet_names:
    print(f"\n=== {sheet} ===")

    df = pd.read_excel(file_path, sheet_name=sheet)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nUnique Type values:")
    print(df["Type"].unique())

    unique_neurons = len(
        set(df["Neuron 1"]).union(
            set(df["Neuron 2"])
        )
    )

    print("\nUnique neurons:")
    print(unique_neurons)

    print("\nConnection counts by type:")
    print(df["Type"].value_counts())

    # ==========================
    # NMJ INSPECTION
    # ==========================

    print("\n" + "=" * 50)
    print("NMJ INSPECTION")
    print("=" * 50)

    nmj = df[df["Type"] == "NMJ"]

    print("\nNumber of NMJ rows:")
    print(len(nmj))

    print("\nFirst 20 NMJ rows:")
    print(nmj.head(20))

    print("\nUnique NMJ source neurons:")
    print(sorted(nmj["Neuron 1"].unique()))

    print("\nUnique NMJ target muscles:")
    print(sorted(nmj["Neuron 2"].unique()))

    print("\nNumber of unique NMJ targets:")
    print(len(nmj["Neuron 2"].unique()))

    print("\nTop NMJ source neurons by connection count:")
    print(nmj["Neuron 1"].value_counts().head(20))

    print("\nTop NMJ target muscles by connection count:")
    print(nmj["Neuron 2"].value_counts().head(20))

    # ==========================
    # SENSORY / MOTOR DISCOVERY
    # ==========================

    print("\n" + "=" * 50)
    print("NEURON NAME INSPECTION")
    print("=" * 50)

    neurons = sorted(
        set(df["Neuron 1"]).union(
            set(df["Neuron 2"])
        )
    )

    print("\nTotal neuron names:")
    print(len(neurons))

    print("\nFirst 100 neuron names:")
    print(neurons[:100])

    print("\nLast 100 neuron names:")
    print(neurons[-100:])

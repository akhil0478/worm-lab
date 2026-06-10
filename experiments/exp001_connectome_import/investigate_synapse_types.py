import pandas as pd

df = pd.read_excel(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

for neuron in ["ADAL", "AVAL", "ASHL"]:
    print("\n" + "=" * 60)
    print("NEURON:", neuron)
    print("=" * 60)

    subset = df[
        (df["Neuron 1"] == neuron)
        | (df["Neuron 2"] == neuron)
    ]

    print(
        subset[
            ["Neuron 1", "Neuron 2", "Type", "Nbr"]
        ].head(50)
    )

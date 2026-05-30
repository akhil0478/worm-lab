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

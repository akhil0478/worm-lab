import pandas as pd

df = pd.read_excel(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

df["Neuron 1"] = (
    df["Neuron 1"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["Neuron 2"] = (
    df["Neuron 2"]
    .astype(str)
    .str.strip()
    .str.upper()
)

s_df = df[df["Type"] == "S"]
r_df = df[df["Type"] == "R"]

print("=" * 60)
print("CHECKING S ↔ R MIRRORING")
print("=" * 60)

matches = 0
total = 0

for _, row in s_df.iterrows():

    total += 1

    src = row["Neuron 1"]
    dst = row["Neuron 2"]
    nbr = row["Nbr"]

    reverse = r_df[
        (r_df["Neuron 1"] == dst)
        & (r_df["Neuron 2"] == src)
        & (r_df["Nbr"] == nbr)
    ]

    if len(reverse) > 0:
        matches += 1

print("S rows:", total)
print("Matching R rows:", matches)

print(
    "Match percentage:",
    round(matches / total * 100, 2),
    "%"
)

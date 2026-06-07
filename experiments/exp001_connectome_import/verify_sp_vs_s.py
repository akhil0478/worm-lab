import pandas as pd

df = pd.read_excel(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

# Normalize names
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
sp_df = df[df["Type"] == "Sp"]

print("=" * 60)
print("COUNTS")
print("=" * 60)

print("S rows :", len(s_df))
print("Sp rows:", len(sp_df))

print("\n")

print("=" * 60)
print("WEIGHT DISTRIBUTIONS")
print("=" * 60)

print("\nS statistics:")
print(s_df["Nbr"].describe())

print("\nSp statistics:")
print(sp_df["Nbr"].describe())

print("\n")

print("=" * 60)
print("OVERLAPPING CONNECTIONS")
print("=" * 60)

s_pairs = set(
    zip(
        s_df["Neuron 1"],
        s_df["Neuron 2"]
    )
)

sp_pairs = set(
    zip(
        sp_df["Neuron 1"],
        sp_df["Neuron 2"]
    )
)

overlap = s_pairs.intersection(sp_pairs)

print("S unique pairs :", len(s_pairs))
print("Sp unique pairs:", len(sp_pairs))
print("Overlap pairs  :", len(overlap))

print("\nFirst 20 overlapping pairs:")
for pair in list(sorted(overlap))[:20]:
    print(pair)

print("\n")

print("=" * 60)
print("EXAMPLES OF MERGED WEIGHTS")
print("=" * 60)

for pair in list(sorted(overlap))[:20]:

    src, dst = pair

    s_weight = (
        s_df[
            (s_df["Neuron 1"] == src)
            & (s_df["Neuron 2"] == dst)
        ]["Nbr"]
        .sum()
    )

    sp_weight = (
        sp_df[
            (sp_df["Neuron 1"] == src)
            & (sp_df["Neuron 2"] == dst)
        ]["Nbr"]
        .sum()
    )

    print(
        f"{src} -> {dst}"
        f" | S={s_weight}"
        f" | Sp={sp_weight}"
        f" | Combined={s_weight + sp_weight}"
    )

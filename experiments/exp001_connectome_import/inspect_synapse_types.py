import pandas as pd

df = pd.read_excel(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

for t in ["S", "R", "Sp", "Rp"]:
    subset = df[df["Type"] == t]

    print(f"\n=== {t} ===")
    print(subset.head(10))

import pandas as pd

df = pd.read_excel(
    "data/raw/connectome_dataset/NeuronConnect1.xls"
)

vc06 = df[
    (df["Neuron 1"] == "VC06")
    | (df["Neuron 2"] == "VC06")
]

print(vc06)

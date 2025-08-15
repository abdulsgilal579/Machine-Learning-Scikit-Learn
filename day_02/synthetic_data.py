from mglearn.datasets import make_wave
import pandas as pd

X, y = make_wave()
df = pd.DataFrame(data=X, columns=["C1"])
df["Target"] = y
print(df.head())


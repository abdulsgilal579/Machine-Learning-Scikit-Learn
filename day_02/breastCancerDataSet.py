from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

data = load_breast_cancer()
print("Cancer Keys {}".format(data.keys()))

#Sample Count Per Class
# print("Sample count per {}".format({n:v for n,v in zip(data.target_names, np.bincount(data.target))}))

#converting in DataFrame
# df = pd.DataFrame(data=data.data, columns=data.feature_names)
# df["Target"] = [data.target_names[t] for t in data.target]
# print(df)
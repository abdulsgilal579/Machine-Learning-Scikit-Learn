from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn
import matplotlib.pyplot as plt

iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)

# irisDataFrame = pd.DataFrame(X_train, columns=iris_dataset["feature_names"])
#
# grr = scatter_matrix(
#     irisDataFrame,
#     c=y_train,
#     figsize=(10, 10),
#     marker="o",
#     hist_kwds={"bins": 20},  # integer number of bins
#     s=60,
#     alpha=0.8,
#     cmap=mglearn.cm3
# )
# plt.show()

knn.fit(X_train, y_train)
# X_new = np.array([[5,2.9,1,0.2]])
# predictions = knn.predict(X_new)
# print(iris_dataset["target_names"][predictions])
y_predict = knn.predict(X_test)
print(knn.score(X_test, y_test))

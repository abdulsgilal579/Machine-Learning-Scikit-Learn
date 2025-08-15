from mglearn.datasets import make_forge
import mglearn
import matplotlib.pyplot as plt
import pandas as pd

X, y = make_forge()



df = pd.DataFrame(data=X, columns=['Feature 0', 'Feature 1'])
df["Target"] = y

#Discrete Scatter Plot - Classification Model
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(['Feature 0', 'Feature 1'], loc='best')
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

#Sythentic Wave Dataset - Rgression Model
X, y = mglearn.datasets.make_wave(n_samples=40)
# X = X[:1]
# y = y[:1]
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
print("X_subset.shape {}".format(X.shape))

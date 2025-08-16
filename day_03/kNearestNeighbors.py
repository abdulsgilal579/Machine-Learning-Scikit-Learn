from mglearn.datasets import make_forge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
clf = KNeighborsClassifier(n_neighbors =3)

# X, y = make_forge()
# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
#
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(f"y_predictions = {y_pred}")
# print(f"accuracy = {clf.score(X_test, y_test)}")

#Evaluating Training and Test Set Performance With Different Number of Neighbors





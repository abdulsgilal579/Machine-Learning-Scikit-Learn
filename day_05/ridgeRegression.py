from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

dataset = fetch_california_housing()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=66)

#Lasso Regression
lasso = Lasso().fit(X_train, y_train)

print(f"Training Set Score: {lasso.score(X_train, y_train)}")
print(f"Test Set Score: {lasso.score(X_test, y_test)}")
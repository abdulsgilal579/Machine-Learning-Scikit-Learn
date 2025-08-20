from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=66)

# Linear Regression
linear_regression = LinearRegression().fit(X_train, y_train)

# Ridge Regression
ridge = Ridge().fit(X_train, y_train)

print("Linear Coefficient: {}".format(linear_regression.coef_))
print("Linear Intercept: {}".format(linear_regression.intercept_))
print("")
print("Linear Regression Train Score: {}".format(linear_regression.score(X_train, y_train)))
print("Linear Regression Test Score: {}".format(linear_regression.score(X_test, y_test)))
print("")
print("Ridge Train Score: {}".format(ridge.score(X_train, y_train)))
print("Ridge Test Score: {}".format(ridge.score(X_test, y_test)))
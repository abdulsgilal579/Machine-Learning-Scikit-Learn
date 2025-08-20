# KNeighborsRegressor Example with California Housing dataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

print("Dataset shape:", X.shape)
print("Feature names:", data.feature_names)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 3. Create and train the model
knn = KNeighborsRegressor(n_neighbors=5, weights="distance")  # try "uniform" too
knn.fit(X_train, y_train)

# 4. Make predictions
y_pred = knn.predict(X_test)

# 5. Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nFirst 5 Predictions:", y_pred[:5])
print("First 5 Actual Values:", y_test[:5])
print("\nMean Squared Error (MSE):", mse)
print("R² Score:", r2)

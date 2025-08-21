# Task 2: House Price Prediction

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 4. Predictions
y_pred = lr_model.predict(X_test)

# 5. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 House Price Prediction (Linear Regression)")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

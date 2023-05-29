import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
california = fetch_california_housing()

# Extract the feature matrix (X) and target vector (y)
X = california.data
y = california.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Get the learned coefficients
coefficients = model.coef_
intercept = model.intercept_

# Print the learned coefficients and intercept
print('Learned coefficients:')
for i, feature_name in enumerate(california.feature_names):
    print(f'{feature_name}: {coefficients[i]}')
print(f'Intercept: {intercept}')

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices (California Housing)')
plt.show()

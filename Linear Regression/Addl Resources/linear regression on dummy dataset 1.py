import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = np.linspace(1, 10, 100).reshape(-1, 1)
y = -2 * X + np.random.randn(100, 1) + 200
import pandas as pd
dicti = {"x":[x[0] for x in X],'y':[Y[0] for Y in y]}
print(dicti)
df = pd.DataFrame(dicti)
df.to_csv("data.csv",index=False)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Plot the data points and the linear regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression')
plt.xlabel('Weight in grams')
plt.ylabel('Height in centimeter')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Print the coefficients and mean squared error
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
print(f'Mean Squared Error: {mse:.2f}')

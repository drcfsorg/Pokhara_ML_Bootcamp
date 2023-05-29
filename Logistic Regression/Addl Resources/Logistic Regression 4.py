import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features (sepal length and sepal width)
y = iris.target  # Target variable (class labels)

# Fit a logistic regression model
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)
model.fit(X, y)

# Create a meshgrid for visualization
h = 0.02  # Step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a color plot of the decision boundaries
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Scatter plot the training data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='r', edgecolors='k', marker='o', label='Setosa')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='g', edgecolors='k', marker='s', label='Versicolor')
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], c='b', edgecolors='k', marker='^', label='Virginica')

# Add labels and legend
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()

# Show the plot
plt.show()

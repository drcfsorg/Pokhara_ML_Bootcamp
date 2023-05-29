import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data  # Features
y = breast_cancer.target  # Target variable (class labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model
model = LogisticRegression(solver='lbfgs',max_iter=3000)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy and create a confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(f"Accuracy of the model is: {accuracy}")

# Plot a histogram of feature values for benign and malignant tumors
features = breast_cancer.feature_names
fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(15, 12))
axes = axes.ravel()

for i in range(len(features)):
    _, bins = np.histogram(X[:, i], bins=30)
    axes[i].hist(X[y == 0][:, i], bins=bins, color='r', alpha=0.5, label='Benign')
    axes[i].hist(X[y == 1][:, i], bins=bins, color='g', alpha=0.5, label='Malignant')
    axes[i].set_title(features[i])
    axes[i].legend()
    axes[i].set_yticks(())

plt.tight_layout()
plt.show()

# Create a heatmap of the confusion matrix
categories = ['Benign','Malignant']
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels = categories, yticklabels = categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

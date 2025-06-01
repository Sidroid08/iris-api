from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

# Load the iris dataset instead of wine
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: 0=setosa, 1=versicolor, 2=virginica

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train logistic regression model
model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Create a directory to save models if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Save both the model and scaler - using .pkl extension as requested
dump(model, 'saved_models/iris_model.pkl')  # Changed to iris_model.pkl
dump(scaler, 'saved_models/iris_scaler.pkl')  # Matching scaler

print("Iris model and scaler saved successfully!")
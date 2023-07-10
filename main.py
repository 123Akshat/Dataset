import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the data from the provided CSV file
data = pd.read_csv('data.csv')

# Select the first hundred lines for training
train_data = data.head(100)

# Select the remaining lines for testing
test_data = data[100:]

# Split the training data into features (X_train) and target (y_train)
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']

# Split the testing data into features (X_test) and target (y_test)
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# Perform one-hot encoding on the feature columns
onehot_encoder = OneHotEncoder(sparse=False)
X_train_encoded = onehot_encoder.fit_transform(X_train)
X_test_encoded = onehot_encoder.transform(X_test)

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train_encoded, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_encoded)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the results to a CSV file
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('malware_detection_results.csv', index=False)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the data from the provided CSV file
data = pd.read_csv('bleh.csv')

# Encode the categorical variable 'Class'
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Perform one-hot encoding on the feature columns
onehot_encoder = OneHotEncoder(sparse=False)
X_encoded = onehot_encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test = X_encoded[:150], X_encoded[150:]
y_train, y_test = y[:150], y[150:]

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the results to a CSV file
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('malware_detection_results.csv', index=False)

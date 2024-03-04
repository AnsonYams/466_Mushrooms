import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 

# Preprocess the data
label_encoder = LabelEncoder()
for column in X.columns:
    X.loc[:, column] = label_encoder.fit_transform(X[column])
for column in y.columns:
    y[column] = label_encoder.fit_transform(y[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = CategoricalNB(alpha=0)

# Train the classifier
model.fit(X_train, y_train)

# Evaluate the classifier
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict labels for the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate accuracy and F1 score for each feature individually
accuracies = []
f1_scores = []
log_losses = []

for i, feature_name in enumerate(X.columns):
    # Create binary feature matrix for the current feature
    # X_test_feature = (X_test.iloc[:, i:i+1] == X_test.iloc[:, i:i+1].unique()[0]).astype(int)
    
    # Evaluate model performance for the current feature
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    log_loss_val = log_loss(y_test, y_pred_proba)
    
    accuracies.append(accuracy)
    f1_scores.append(f1)
    log_losses.append(log_loss_val)
    
    print(f"Feature: {feature_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {log_loss_val:.4f}")
    print()

# Overall model performance
print("Overall Model Performance:")
print(f"Accuracy: {np.mean(accuracies)}")
print(f"F1 Score: {np.mean(f1_scores)}")
print(f"Log Loss: {np.mean(log_losses)}")
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from p_decision_tree.DecisionTree import DecisionTree
from ucimlrepo import fetch_ucirepo

# Fetch dataset
mushroom = fetch_ucirepo(id=73)

# Data (as pandas dataframes)
X = pd.DataFrame(mushroom.data.features)
y = pd.DataFrame(mushroom.data.targets)

# Combine features and target into a single DataFrame
data = pd.concat([X, y], axis=1)
data = data.dropna()
# Converting all the columns to string, if not already
data = data.map(str)

# Splitting the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extract training data
train_descriptive = train_data.iloc[:, :-1].values.tolist()
train_label = train_data.iloc[:, -1].values.tolist()

# Extract test data
test_descriptive = test_data.iloc[:, :-1].values.tolist()
test_label = test_data.iloc[:, -1].values.tolist()

# Initialize the DecisionTree
decisionTree = DecisionTree(
    train_descriptive, data.columns[:-1].tolist(), train_label, "entropy")

# Build the tree
decisionTree.id3(0, 0)

# Visualize the decision tree 
dot = decisionTree.print_visualTree(
    render=True)  # Set render=True to visualize

# Implement the predict and calculate_accuracy methods in DecisionTree class beforehand

# Calculate accuracy on the training set
train_accuracy = decisionTree.calculate_accuracy(
    train_descriptive, train_label)
print(f"Accuracy on the training set: {train_accuracy * 100:.2f}%")

# Calculate accuracy on the testing set
test_accuracy = decisionTree.calculate_accuracy(test_descriptive, test_label)
print(f"Accuracy on the test set: {test_accuracy * 100:.2f}%")

# System entropy
print("System entropy:", decisionTree.entropy)
# print(data)

# # Predict outcomes for the test set
# test_predictions = [decisionTree.predict(
#     decisionTree.root, sample) for sample in data.iloc[:, :-1].values.tolist()]

# # Calculate the confusion matrix
# conf_matrix = confusion_matrix(
#     data.iloc[:, -1].values.tolist(), test_predictions, labels=decisionTree.labelCodes)

# print("Confusion Matrix:")
# print(conf_matrix)
# print(data['poisonous'].value_counts())



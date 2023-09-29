# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:07:10 2023

@author: arneg
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def entropy(y):
    # Calculate entropy for a binary classification problem
    p = np.mean(y)  # Probability of class 1
    if p == 0 or p == 1:
        return 0  # Entropy is 0 for pure labels
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def information_gain(y, left_y, right_y):
    # Calculate information gain for a split
    H = entropy(y)
    H_left = entropy(left_y)
    H_right = entropy(right_y)
    p_left = len(left_y) / len(y)
    p_right = len(right_y) / len(y)
    gain = H - (p_left * H_left + p_right * H_right)
    return gain

def information_gain_ratio(y, left_y, right_y):
    # Calculate information gain ratio for a split
    gain = information_gain(y, left_y, right_y)
    H_left = entropy(left_y)
    H_right = entropy(right_y)
    
    if H_left + H_right == 0:
        return 0
    
    ratio = gain / (H_left + H_right)
    return ratio

def decision_tree(X, y, depth=0):
    n_samples, n_features = X.shape
    
    y = y.astype(int)
    
    best_split = None
    best_gain_ratio = -1
    
    for feature_idx in range(n_features):
        unique_values = np.unique(X[:, feature_idx])
        unique_values.sort()
    
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            left_mask = X[:, feature_idx] < threshold
            right_mask = X[:, feature_idx] >= threshold
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            if len(left_y) == 0 or len(right_y) == 0:
                continue
    
            gain_ratio = information_gain_ratio(y, left_y, right_y)
            
            #print(f"Split on feature {feature_idx}, threshold {threshold}: Gain Ratio = {gain_ratio:.4f}")
            
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_split = (feature_idx, threshold)
    
    if best_gain_ratio == 0:
        return 1
    
    if best_split is None:
        return 1
    
    feature_idx, threshold = best_split
    left_mask = X[:, feature_idx] < threshold
    right_mask = X[:, feature_idx] >= threshold
    left_y = y[left_mask]
    right_y = y[right_mask]
    
    if len(left_y) == 0 or len(right_y) == 0:
        return 1
    
    left_subtree = decision_tree(X[left_mask], left_y, depth + 1)
    right_subtree = decision_tree(X[right_mask], right_y, depth + 1)
    
    print("Best Split:", best_split)
    
    return (feature_idx, threshold, left_subtree, right_subtree)




def count_nodes(tree):
    if isinstance(tree, int):
        return 1
    elif isinstance(tree, tuple):
        return 1 + count_nodes(tree[2]) + count_nodes(tree[3])
    else:
        raise ValueError("Invalid tree structure")
    
def predict_tree(tree, x):
    if isinstance(tree, int):
        return tree
    feature_idx, threshold, left_subtree, right_subtree = tree
    if x[feature_idx] < threshold:
        return predict_tree(left_subtree, x)
    else:
        return predict_tree(right_subtree, x)
    
def predict(tree, X):
    return [predict_tree(tree, x) for x in X]

def calculate_test_error(tree, X_test, y_test):
    y_pred = predict(tree, X_test)
    accuracy = np.mean(y_pred == y_test)
    return 1 - accuracy


#########################
# Q2.3
##########################

path = 'Homework 2 data\Druns.txt'
data = np.genfromtxt(path, delimiter=' ')

X = data[:, :-1]
y = data[:, -1]

druns_tree = decision_tree(X,y)


#########################
# Q2.4
##########################

path = 'Homework 2 data\D3leaves.txt'
data = np.genfromtxt(path, delimiter=' ')

X = data[:, :-1]
y = data[:, -1]

d3_tree = decision_tree(X,y)


#                [Feature x1 <= 5.5]
#                /                \
#      [Feature X2 <= 1.5]        [Class 1]
#         /          \           
#     [Class 0]    [Class 1]   


#########################
# Q2.5
##########################

path = 'Homework 2 data\D1.txt'
d1= np.genfromtxt(path, delimiter=' ')

X = d1[:, :-1]
y = d1[:, -1]

d1_tree = decision_tree(X,y)



#                [Feature x2 <= 0.2]
#                /                \
#            [Class 0]        [Class 1]

path = 'Homework 2 data\D2.txt'
d2 = np.genfromtxt(path, delimiter=' ')

X = d2[:, :-1]
y = d2[:, -1]

d2_tree = decision_tree(X,y)

#                [Feature x1 <= 0.5]
#                /                \
#            [Class 0]        [Class 1]


#########################
# Q2.7
##########################

import random

path = 'Homework 2 data\Dbig.txt'
dbig = np.genfromtxt(path, delimiter=' ')

random.shuffle(dbig)

training_size = 8192

training_set = dbig[:training_size]
test_set = dbig[training_size:]


# Define the sizes of the nested training sets
training_sizes = [32, 128, 512, 2048, 8192]

# Initialize an empty list to store the nested training sets
nested_training = []

# Create the nested training sets
for size in training_sizes:
    nested_set = dbig[:size]  
    nested_training.append(nested_set)
    
    X_train = dbig[:size][:, :-1]
    y_train = dbig[:size][:, -1]
    
    X_test = dbig[size:][:, :-1]
    y_test = dbig[size:][:, -1]
    
    # call decision tree function
    tree = decision_tree(X_train,y_train)
    
    num_nodes = count_nodes(tree)
    print(f"Number of nodes in the tree: {num_nodes}")

    test_error = calculate_test_error(tree, X_test, y_test)
    print(f"Test set error: {test_error:.4f}")
    
    






##########################################
#  Q3
##########################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier()


# Define the sizes of the nested training sets
training_sizes = [32, 128, 512, 2048, 8192]

# Initialize an empty list to store the nested training sets
nested_training = []

# Initialize lists to store results
node_counts = []
test_errors = []
n_ = []


# Create the nested training sets
for size in training_sizes:
    nested_set = dbig[:size]  
    nested_training.append(nested_set)
    
    X_train = dbig[:size][:, :-1]
    y_train = dbig[:size][:, -1]
    
    X_test = dbig[size:][:, :-1]
    y_test = dbig[size:][:, -1]
    
    clf.fit(X_train, y_train)  
    
    nodes = clf.tree_.node_count 
    
    y_pred = clf.predict(X_test)
    
    err = 1 - accuracy_score(y_test, y_pred)
    
    n_.append(len(dbig[:size]))
    node_counts.append(nodes)
    test_errors.append(err)
    
plt.figure(figsize=(10, 6))
plt.plot(node_counts, test_errors, marker='o', linestyle='-', color='b')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Test Set Error (errn)')
plt.title('Learning Curve')
plt.grid(True)
plt.show()

print(f'n: {n_}')
print(f'node counts: {node_counts}')
print(f'test errors: {test_errors}')








# Define the range for your feature space
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Create a meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Tree Decision Boundary')
plt.show()




##########################################
#  Q4
##########################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier()

# Define the sizes of the nested training sets
data = np.random.uniform(0, 300, 100)
y_values = np.sin(data)
df = pd.DataFrame({'x': data, 'y': y_values})

X_train = df[['x']]
y_train = df['y']

test_data = np.random.uniform(0, 300, 100)
test_y_values = np.sin(data)
test_df = pd.DataFrame({'x': test_data, 'y': test_y_values})

X_test = test_df[['x']]
y_test = test_df['y']

    
clf.fit(X_train, y_train)  
    
# Vary the standard deviation for epsilon
noise_std_dev_values = [0.1, 0.5, 1.0, 2.0]

for std_dev in noise_std_dev_values:
    # Generate noisy test set
    noisy_X_test = X_test + np.random.normal(0, std_dev, size=X_test.shape)
    
    # Use the trained model to predict labels for the noisy test set
    y_test_pred = clf.predict(noisy_X_test)
    
    # Calculate test error on the noisy test set
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    
    print(f"Noise Std Dev: {std_dev}, Test Error: {test_error:.4f}")


##########################################
#  scatter plot
##########################################


x1 = [1,2,3,4]
x2 = [1,2,3,4]
y = [0,1,0,1]

df = pd.DataFrame({'x1':np.array(x1),'x2':np.array(x2),'y':np.array(y)})

plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='jet', marker='o', label='class')

# Create separate DataFrames for each class
class_0 = df[df['y'] == 0]
class_1 = df[df['y'] == 1]

# Create a scatter plot for each class with a custom label
plt.scatter(class_0['x1'], class_0['x2'], c='blue', marker='o', label='Class 0')
plt.scatter(class_1['x1'], class_1['x2'], c='red', marker='o', label='Class 1')

# Create a scatter plot for the entire dataset with no label
plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='jet', marker='o', label='')


plt.xlabel('x1')
plt.ylabel('x2')

# Show the legend with customized handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
# Exclude the first item in handles and labels (the empty label)
handles = handles[1:]
labels = labels[1:]
plt.legend(handles=handles, labels=labels)

# Display the plot
plt.show()

#############################


column_names = ['x1','x2','y']
df = pd.DataFrame(data, columns=column_names)

plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='jet', marker='o', label='class')

# Create separate DataFrames for each class
class_0 = df[df['y'] == 0]
class_1 = df[df['y'] == 1]

# Create a scatter plot for each class with a custom label
plt.scatter(class_0['x1'], class_0['x2'], c='blue', marker='o', label='Class 0')
plt.scatter(class_1['x1'], class_1['x2'], c='red', marker='o', label='Class 1')

# Create a scatter plot for the entire dataset with no label
plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='jet', marker='o', label='')


plt.xlabel('x1')
plt.ylabel('x2')

# Show the legend with customized handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
# Exclude the first item in handles and labels (the empty label)
handles = handles[1:]
labels = labels[1:]
plt.legend(handles=handles, labels=labels)

# Display the plot
plt.show()

###############################

path = 'Homework 2 data\Dbig.txt'
dbig = np.genfromtxt(path, delimiter=' ')

random.shuffle(dbig)

training_size = 8192

training_set = dbig[:training_size]
test_set = dbig[training_size:]


# Define the sizes of the nested training sets
training_sizes = [32, 128, 512, 2048, 8192]

# Initialize an empty list to store the nested training sets
nested_training = []

# Create the nested training sets
for size in training_sizes:
    nested_set = dbig[:size]  
    nested_training.append(nested_set)
    
    X_train = dbig[:size][:, :-1]
    y_train = dbig[:size][:, -1]
    
    X_test = dbig[size:][:, :-1]
    y_test = dbig[size:][:, -1]
    
    # call decision tree function
    tree = decision_tree(X_train,y_train)
    
    num_nodes = count_nodes(tree)
    print(f"Number of nodes in the tree: {num_nodes}")

    test_error = calculate_test_error(tree, X_test, y_test)
    print(f"Test set error: {test_error:.4f}")

    
    X_train_df = pd.DataFrame(X_train, columns=['x1', 'x2'])
    y_train_df = pd.DataFrame(y_train, columns=['y'])
    
    # Combine X_train_df and y_train_df into a single DataFrame
    df = pd.concat([X_train_df, y_train_df], axis=1)
    
    plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='jet', marker='o', label='class')
    
    # Create separate DataFrames for each class
    class_0 = df[df['y'] == 0]
    class_1 = df[df['y'] == 1]
    
    # Create a scatter plot for each class with a custom label
    plt.scatter(class_0['x1'], class_0['x2'], c='blue', marker='o', label='Class 0')
    plt.scatter(class_1['x1'], class_1['x2'], c='red', marker='o', label='Class 1')
    
    # Create a scatter plot for the entire dataset with no label
    plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='jet', marker='o', label='')
    
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # Show the legend with customized handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Exclude the first item in handles and labels (the empty label)
    handles = handles[1:]
    labels = labels[1:]
    plt.legend(handles=handles, labels=labels)
    
    # Display the plot
    plt.show()
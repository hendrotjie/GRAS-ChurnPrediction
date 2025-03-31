import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def F1(indi, df, clf):
    """ Fitness function for selected features """
    thr = 0.5
    x = np.asarray(indi).reshape(1, -1)
    
    # Binarize the solution vector
    x[x >= thr] = 1
    x[x < thr] = 0

    # Identify selected features
    selected_features = np.where(x == 1)[1]  # This line was missing before

    # If no features are selected, select the first feature by default
    if len(selected_features) == 0:
        selected_features = np.array([0])

    # Create the subset of the dataset based on the selected features
    X = df.iloc[:, selected_features].values
    y = df.iloc[:, -1].values  # Assuming the target column is the last one
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Use KNN classifier with k=5 and Euclidean distance
    #clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
       
    
    # Calculate fitness based on both accuracy and the number of selected features
    error_rate = 1 - accuracy
    num_selected_features = len(selected_features)
    total_num_features = df.shape[1] - 1  # minus the label column
    normalized_feature_count = num_selected_features / total_num_features

    # Custom fitness function
    alpha = 0.99  # Weighting factor for accuracy
    beta = 1 - alpha  # Weighting factor for the number of features
    fitness = (alpha * error_rate) + (beta * normalized_feature_count)
    
    return fitness

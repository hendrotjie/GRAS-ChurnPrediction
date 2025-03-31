import pandas as pd
import time 
import numpy as np
import random
from datetime import datetime as dt
from utils import train_model
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



def get_classifier(classifier_name):
    if classifier_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    elif classifier_name in ["SVM", "Support Vector Machine"]:
        return SVC(probability=True)  # Adding probability=True for ROC-AUC compatibility
    elif classifier_name in ["RandomForest", "Random Forest"]:
        return RandomForestClassifier(random_state=42)
    elif classifier_name in ["LogisticRegression", "Logistic Regression"]:
        return LogisticRegression(random_state=42)
    elif classifier_name in ["NaiveBayes", "Naive Bayes"]:
        return GaussianNB()
    elif classifier_name in ["DecisionTree", "Decision Tree"]:
        return DecisionTreeClassifier(random_state=42)
    elif classifier_name in ["GradientBoosting", "Gradient Boosting"]:
        return GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")


# Get the directory of the current script or notebook
script_dir = os.path.dirname(os.path.abspath("__file__"))

# Construct the full path to the 'src' directory
src_dir = os.path.join(script_dir, '..', 'src')

# Append the 'src' directory to sys.path
sys.path.append(src_dir)

# Construct the full path to the 'data' directory
data_dir = os.path.join(script_dir, '..', 'data')

# Define paths
DATA_PATH = '../data/processed/'
OUTPUT_PATH = '../results/'

# Load data from churn2.csv
#df = pd.read_csv(r'C:\Users\fakep\Documents\S3 ITS\Machine Learning with Python\Simulated-Annealing-Feature-Selection-main\data\churn2.csv')

#df = pd.read_csv(r'C:\Users\hendr\Documents\S3 ITS\Machine Learning with Python\MachineLearning\Simulated-Annealing-Feature-Selection-main\data\churn2.csv')

#df = pd.read_csv(os.path.join(data_dir, 'churn2.csv'))


# Drop the first column as it's not used
#df = df.iloc[:, 1:]

# Separate features and target
#X_train = df.drop(columns=['churn'])
#y_train = df['churn']

# Setup simulated annealing algorithm
# def simulated_annealing(X_train, y_train, classifier_name="KNN", fitness_function="accuracy", run_index=1, maxiters=50, alpha=0.93, beta=1, T_0=1, update_iters=1, temp_reduction='geometric'):
# #     columns = ['Iteration', 'Feature Count', 'Feature Set', 'Accuracy','Best Accuracy',
# #                'Acceptance Probability', 'Random Number', 'Outcome',
# #                'Precision', 'Recall', 'F1 Score', 'AUC Score',
# #                'Best Precision', 'Best Recall', 'Best F1 Score', 'Best AUC Score',
# #                'Run Time (s)']

#menggunakan fitness function baru
def simulated_annealing(X_train, y_train, initial_solution=None, classifier_name="KNN", fitness_function="custom_fitness", run_index=1, maxiters=50, alpha=0.93, beta_value=0.01, beta=1, update_iters=1, temp_reduction='geometric'):
    columns = ['Iteration', 'Feature Count', 'Feature Set', 
               'Acceptance Probability', 'Random Number', 'Outcome',
               'Fitness', 'Best Fitness', 'Run Time (s)',
               'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score',
               'Best Accuracy', 'Best Precision', 'Best Recall', 'Best F1 Score', 'Best AUC Score']

    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_subset = None
    hash_values = set()
    
    
    # Set initial temperature based on the number of features
    T_0 = 2 * X_train.shape[1]  # Initial Temperature = 2 * |N|
    T = T_0  # Start with initial temperature

     
    #T = T_0

    # Get ascending range indices of all columns
    full_set = set(np.arange(len(X_train.columns)))

    # Use the initial solution if provided, otherwise generate a random subset
    if initial_solution is not None:
        curr_subset = set(initial_solution)
    else:
        curr_subset = set(random.sample(list(full_set), round(0.5 * len(full_set))))
    
    # Get the classifier object based on the name
    clf = get_classifier(classifier_name)
    
    # Generate initial random subset based on ~50% of columns
    curr_subset = set(random.sample(list(full_set), round(0.5 * len(full_set))))

    # Get baseline metric score (i.e. AUC) of initial random subset
    X_curr = X_train.iloc[:, list(curr_subset)]
    
    # Get baseline metric from new fitness function
    prev_metric = train_model(X_curr, y_train, clf, solution)
    
    # Calculate the initial fitness
    er = 1 - prev_metric['accuracy']
    m = len(curr_subset)
    N = len(full_set)
    fitness = (0.99 * er) + (beta_value * (m / N))
    best_fitness = fitness
    
    #prev_metric = train_model(X_curr, y_train)
    best_metric = prev_metric
    best_subset = curr_subset.copy()  # Initialize best_subset

    for i in range(maxiters):
        
        #start timing the iteration
        start_time = time.time()
        
        if T < 0.01:
            print(f'Temperature {T} below threshold. Termination condition met')
            break
        
        print(f'Starting Iteration {i+1}')

        while True:
            if len(curr_subset) == len(full_set): 
                move = 'Remove'
            elif len(curr_subset) == 2:
                move = random.choice(['Add', 'Replace'])
            else:
                move = random.choice(['Add', 'Replace', 'Remove'])
            
            pending_cols = full_set.difference(curr_subset) 
            new_subset = curr_subset.copy()   

            if move == 'Add':        
                new_subset.add(random.choice(list(pending_cols)))
            elif move == 'Replace': 
                new_subset.remove(random.choice(list(curr_subset)))
                new_subset.add(random.choice(list(pending_cols)))
            else:
                new_subset.remove(random.choice(list(curr_subset)))
                
            if new_subset in hash_values:
                print('Subset already visited')
            else:
                hash_values.add(frozenset(new_subset))
                break

        X_new = X_train.iloc[:, list(new_subset)]
        metric = train_model(X_new, y_train, clf)
        
        # Calculate the fitness for the new subset
        er = 1 - metric['accuracy']
        m = len(new_subset)
        fitness = (0.99 * er) + (beta_value * (m / N))
        
        if fitness < best_fitness:  # We want to minimize the fitness
            print(f'Improvement in Fitness from {best_fitness:.4f} to {fitness:.4f} - New subset accepted')
            outcome = 'Improved'
            accept_prob, rnd = '-', '-'
            best_fitness = fitness
            best_metric = metric
            best_subset = new_subset.copy()
            curr_subset = new_subset.copy()
        else:
            rnd = np.random.uniform()
            diff = fitness - best_fitness
            accept_prob = np.exp(-beta * diff / T)
            
            if rnd < accept_prob:
                print(f'Worse Fitness but accepted. Fitness change: {diff:.4f}, Acceptance probability: {accept_prob:.4f}, Random number: {rnd:.4f}')

                outcome = 'Accept'
                curr_subset = new_subset.copy()
            else:
                print(f'Worse Fitness and rejected. Fitness change: {diff:.4f}, Acceptance probability: {accept_prob:.4f}, Random number: {rnd:.4f}')
                outcome = 'Reject'

        end_time = time.time()
        iteration_time = end_time - start_time

        # Update results dataframe
        results.loc[i, 'Iteration'] = i+1
        results.loc[i, 'Feature Count'] = len(curr_subset)
        results.loc[i, 'Feature Set'] = sorted(curr_subset)
        results.loc[i, 'Fitness'] = fitness
        results.loc[i, 'Best Fitness'] = best_fitness
        results.loc[i, 'Acceptance Probability'] = accept_prob
        results.loc[i, 'Random Number'] = rnd
        results.loc[i, 'Outcome'] = outcome
        results.loc[i, 'Accuracy'] = metric.get('accuracy')
        results.loc[i, 'Precision'] = metric.get('precision')
        results.loc[i, 'Recall'] = metric.get('recall')
        results.loc[i, 'F1 Score'] = metric.get('f1')
        results.loc[i, 'AUC Score'] = metric.get('auc_score', np.nan)
        results.loc[i, 'Best Accuracy'] = best_metric.get('accuracy')
        results.loc[i, 'Best Precision'] = best_metric.get('precision')
        results.loc[i, 'Best Recall'] = best_metric.get('recall')
        results.loc[i, 'Best F1 Score'] = best_metric.get('f1')
        results.loc[i, 'Best AUC Score'] = best_metric.get('auc_score', np.nan)
        results.loc[i, 'Run Time (s)'] = iteration_time  # Store the run time for this iteration
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         if metric['accuracy'] > prev_metric['accuracy']:
#             print('Local improvement in Accuracy from {:8.4f} to {:8.4f}'
#                   .format(prev_metric['accuracy'], metric['accuracy']) + ' - New subset accepted')
#             outcome = 'Improved'
#             accept_prob, rnd = '-', '-'
#             prev_metric = metric
#             curr_subset = new_subset.copy()

#             if metric['accuracy'] > best_metric['accuracy']:
#                 print('Global improvement in Accuracy from {:8.4f} to {:8.4f}'
#                       .format(best_metric['accuracy'], metric['accuracy']) + ' - Best subset updated')
#                 best_metric = metric
#                 best_subset = new_subset.copy()    
                
#         else:
#             rnd = np.random.uniform()
#             diff = prev_metric['accuracy'] - metric['accuracy']
#             accept_prob = np.exp(-beta * diff / T)

#             if rnd < accept_prob:
#                 print('New subset has worse performance but still accept. Metric change' +
#                       ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
#                       .format(diff, accept_prob, rnd))
#                 outcome = 'Accept'
#                 prev_metric = metric
#                 curr_subset = new_subset.copy()
#             else:
#                 print('New subset has worse performance, therefore reject. Metric change' +
#                       ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
#                       .format(diff, accept_prob, rnd))
#                 outcome = 'Reject'

#         # End timing the iteration
#         end_time = time.time()
#         iteration_time = end_time - start_time
                
                
#         # Update results dataframe
#         results.loc[i, 'Iteration'] = i+1
#         results.loc[i, 'Feature Count'] = len(curr_subset)
#         results.loc[i, 'Feature Set'] = sorted(curr_subset)
#         results.loc[i, 'Acceptance Probability'] = accept_prob
#         results.loc[i, 'Random Number'] = rnd
#         results.loc[i, 'Outcome'] = outcome
#         results.loc[i, 'Accuracy'] = metric.get('accuracy')
#         results.loc[i, 'Precision'] = metric.get('precision')
#         results.loc[i, 'Recall'] = metric.get('recall')
#         results.loc[i, 'F1 Score'] = metric.get('f1')
#         results.loc[i, 'AUC Score'] = metric.get('auc_score', np.nan)
#         results.loc[i, 'Best Accuracy'] = best_metric.get('accuracy')
#         results.loc[i, 'Best Precision'] = best_metric.get('precision')
#         results.loc[i, 'Best Recall'] = best_metric.get('recall')
#         results.loc[i, 'Best F1 Score'] = best_metric.get('f1')
#         results.loc[i, 'Best AUC Score'] = best_metric.get('auc_score', np.nan)
#         results.loc[i, 'Run Time (s)'] = iteration_time 
        
        if i % update_iters == 0:
            if temp_reduction == 'geometric':
                T = alpha * T #Geometric cooling with alpha = 0.93
            elif temp_reduction == 'linear':
                T -= alpha
            elif temp_reduction == 'slow decrease':
                b = 5
                T = T / (1 + b * T)
            else:
                raise Exception("Temperature reduction strategy not recognized")

    if best_subset is None:
        best_subset = curr_subset  # Fallback to current subset if no best found            
    
    best_subset_cols = [list(X_train.columns)[i] for i in list(best_subset)]
    results = results.dropna(axis=0, how='all')
    
    # Custom filename with classifier, fitness function, run index, and time
    dt_string = f"{classifier_name}_{fitness_function}_Run{run_index}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    
    #dt_string = dt.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(f'{OUTPUT_PATH}/sa_output_{dt_string}.csv', index=False)

    return results, best_metric, best_subset_cols

#if __name__ == '__main__':
#    results, best_metric, best_subset_cols = simulated_annealing(X_train, y_train)


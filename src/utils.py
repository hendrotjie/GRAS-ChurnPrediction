import os
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Intel MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # NumExpr threads
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # OpenBLAS threads


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature selection based on binary solutions.
    """
    def __init__(self, solution):
        self.solution = solution

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Select features based on binary solution
        selected_features = [i for i, bit in enumerate(self.solution) if bit >= 0.5]
        if not selected_features:  # Ensure at least one feature is selected
            selected_features = [0]  # Select at least one feature
        print(f"Selected features: {selected_features}")  # Debug output
        return X.iloc[:, selected_features]


def get_classifier(clf_name):
    """
    Returns a classifier object based on the input name with regularization parameters.
    """
    if clf_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, metric='euclidean')  # Keep this simple
    elif clf_name in ["SVM", "Support Vector Machine"]:
        # Regularization for SVM
        return SVC(probability=True, C=5.0, kernel='linear', class_weight='balanced', random_state=42)  # C is the regularization parameter
    elif clf_name in ["RandomForest", "Random Forest"]:
        # Add max_depth and min_samples_split for regularization
        return RandomForestClassifier(
            random_state=42,
            max_depth=3,  # Limits depth of the tree
            min_samples_split=30,  # Minimum samples required to split an internal node
            n_estimators=50
        )
    elif clf_name in ["LogisticRegression", "Logistic Regression"]:
        # Add L2 regularization (default in Logistic Regression)
        return LogisticRegression(random_state=42, penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    elif clf_name in ["NaiveBayes", "Naive Bayes"]:
        return GaussianNB()  # No direct regularization needed for Naive Bayes
    elif clf_name in ["DecisionTree", "Decision Tree"]:
        # Add regularization through max_depth and min_samples_split
        return DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=10)
    elif clf_name in ["GradientBoosting", "Gradient Boosting"]:
        # Add learning rate and max depth for regularization
        return GradientBoostingClassifier(
            random_state=42,
            max_depth=3,  # Shallow trees for regularization
            learning_rate=0.1,  # Step size shrinkage
            n_estimators=100  # Number of boosting stages
        )
    else:
        raise ValueError(f"Unsupported classifier: {clf_name}")



def train_model(X_train, y_train, clf, solution):
    """
    Train and evaluate a classification model using cross-validation on the training set.
    """
    # Create a pipeline with feature selection and classification
    pipeline = Pipeline([
        ('feature_selection', FeatureSelector(solution=solution)),  # Feature selection
        ('classifier', clf)  # Classifier
    ])

    # Use cross-validation on the training data
    kf = KFold(shuffle=True, n_splits=3, random_state=42)  # 5-fold cross-validation

    accuracy = cross_val_score(pipeline, X_train, y_train.values.ravel(), cv=kf, scoring='accuracy', n_jobs=4).mean()
    precision = cross_val_score(pipeline, X_train, y_train.values.ravel(), cv=kf, scoring='precision', n_jobs=4).mean()
    recall = cross_val_score(pipeline, X_train, y_train.values.ravel(), cv=kf, scoring='recall', n_jobs=4).mean()
    f1 = cross_val_score(pipeline, X_train, y_train.values.ravel(), cv=kf, scoring='f1', n_jobs=4).mean()
    auc = cross_val_score(pipeline, X_train, y_train.values.ravel(), cv=kf, scoring='roc_auc', n_jobs=4).mean()

    metrics = {
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3),
        'auc': round(auc, 3)
    }
    return metrics


def fitness_function(solution, X_train, y_train, clf=None, alpha=0.99):
    """
    Custom fitness function that evaluates a solution by combining error rate and feature count.
    `alpha` controls the trade-off between accuracy and subset size (0 < alpha < 1).
    """
    # Convert solution to binary to identify selected features
    num_selected_features = sum([bit >= 0.5 for bit in solution])
    if num_selected_features == 0:
        num_selected_features = 1  # Ensure at least one feature is selected

    # Use get_classifier if clf is a string
    if isinstance(clf, str):
        clf = get_classifier(clf)

    # Evaluate model performance
    metrics = train_model(X_train, y_train, clf, solution)
    error_rate = 1 - metrics['accuracy']  # Error rate is 1 - accuracy

    # Feature count and normalization
    total_features = X_train.shape[1]
    normalized_feature_count = num_selected_features / total_features

    # Calculate fitness
    beta = 1 - alpha
    fitness = (alpha * error_rate) + (beta * normalized_feature_count)
    return fitness

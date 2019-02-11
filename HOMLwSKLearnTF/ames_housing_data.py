# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:17:32 2018

@author: Chris McIntyre

Script is to analyze the Ames Real Estate dataset from Kaggle website.
"""

# Import packages
import pandas as pd
from helpers import preprocess_mixed_data
from helpers import load_data
from helpers import get_regression_estimator
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestRegressor

#load the data form CSV
train_data = load_data("train.csv", "datasets\\ames_data\\")
test_data = load_data("test.csv", "datasets\\ames_data\\")   #   Used for final submission

y_train, X_train = preprocess_mixed_data(train_data, "median", "ZZ", "SalePrice", True)
X_test_final = preprocess_mixed_data(test_data, "median", "ZZ", "", True)

split = int(len(y_train)*.9)
y_test = y_train[split:]
X_test = X_train[split:]

y_train = y_train[:split]
X_train = X_train[:split]
"""
#   Looks for correlations in the data
attributes = train_data.keys()
pd.scatter_matrix(train_data[attributes[65:80]], figsize=(20,12))
plt.show()
""" 
###########################################################
####    Algorithm Development Section          ############
#            Alorithms to try 
#            ensemble:
#                AdaBoostRegressor;
#                BaggingRegressor;
#                ExtraTreesRegressor;
#                GradientBoostingRegressor;
#                RandomForestRegressor;
#            neighbors:
#                KNeighborsRegressor;
#                RadiusNeighborsRegressor;
#            neural_network:
#                MLPRegressor;
#            svm:
#                LinearSVR;
#                NuSVR;
#                SVR;
#            tree:
#                DecisionTreeRegressor;
#                ExtraTreeRegressor;
###########################################################

# use a full grid over all estimators
estimators = ["RandomForestRegressor",
              "AdaBoostRegressor",
              "BaggingRegressor",
              "ExtraTreesRegressor",
              "GradientBoostingRegressor",
              "KNeighborsRegressor",
              "RadiusNeighborsRegressor",
              "MLPRegressor",
              "LinearSVR",
              "NuSVR",
              "SVR",
              "DecisionTreeRegressor",
              "ExtraTreeRegressor",
              "KNeighborsRegressor",
              "RadiusNeighborsRegressor",
              "MLPRegressor",
              "LinearSVR",
              "NuSVR",
              "SVR",
              "DecisionTreeRegressor",
              "ExtraTreeRegressor"]


best_est_score = 0
best_est = RandomForestRegressor()
best_est_name = ""
estimator_highlights = []
for item in estimators:
    
    param_grid, estimator_ = get_regression_estimator(item)
            
    # run grid search over all estimators and paramters
    grid_search = GridSearchCV(estimator_, param_grid=param_grid, refit=True, cv=5, verbose= 3)
    grid_search.fit(X_train, y_train)
    grid_search.predict(X_test)
    score = grid_search.score(X_test, y_test)

    if score > best_est_score:
        best_est_score = score
        best_est = grid_search.best_estimator_
        best_est_name = item
    
    #save the score of this estimator
    estimator_highlights.append(item)
    estimator_highlights.append(grid_search.best_score_)
    
train_sizes, train_scores, test_scores = learning_curve(best_est, 
                                                        X_train, 
                                                        y_train, 
                                                        cv=5,
                                                        train_sizes=np.linspace(0.1, 1.0, 7),
                                                        verbose= 3)
"""
        estimator : object type that implements the “fit” and “predict” methods
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples) or (n_samples, n_features), optional
        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to generate the learning curve. If the dtype is float, it is regarded as a fraction of the maximum size of the training set (that is determined by the selected validation method), i.e. it has to be within (0, 1]. Otherwise it is interpreted as absolute sizes of the training sets. Note that for classification the number of samples usually have to be big enough to contain at least one sample from each class. (default: np.linspace(0.1, 1.0, 5))
        cv : int, cross-validation generator or an iterable, optional
        scoring : string, callable or None, optional, default: None
            A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
            exploit_incremental_learning : boolean, optional, default: False
            If the estimator supports incremental learning, this will be used to speed up fitting for different training set sizes.
        n_jobs : integer, optional
        pre_dispatch : integer or string, optional
            Number of predispatched jobs for parallel execution (default is all). The option can reduce the allocated memory. The string can be an expression like ‘2*n_jobs’.
        verbose : integer, optional
        shuffle : boolean, optional
"""
        
plt.figure()
plt.title(best_est_name)
plt.ylim(.65, 1.1)
plt.xlabel("Training examples")
plt.ylabel("Score")

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()
print("Best Estimator: ", best_est_name, "\tScore: ", best_est_score)
print(estimator_highlights)

"""
REGRESSION performance Metrics:
    metrics.explained_variance_score(y_true, y_pred) 
        Explained variance regression score function 
    metrics.mean_absolute_error(y_true, y_pred) 
        Mean absolute error regression loss 
    metrics.mean_squared_error(y_true, y_pred[, …]) 
        Mean squared error regression loss 
    metrics.mean_squared_log_error(y_true, y_pred) 
        Mean squared logarithmic error regression loss 
    metrics.median_absolute_error(y_true, y_pred) 
        Median absolute error regression loss 
    metrics.r2_score(y_true, y_pred[, …]) 
        R^2 (coefficient of determination) regression score function. 
        
Utilities to evaluate models with respect to a variable
    learning_curve.learning_curve(estimator, X, y) 
        Learning curve. 
    learning_curve.validation_curve(estimator, ...) 
        Validation curve.
        
        EXAMPLE CURVE PLOTTING:
            print(__doc__)
    
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.datasets import load_digits
            from sklearn.svm import SVC
            from sklearn.learning_curve import validation_curve
            
            digits = load_digits()
            X, y = digits.data, digits.target
            
            param_range = np.logspace(-6, -1, 5)
            train_scores, test_scores = validation_curve(
                SVC(), X, y, param_name="gamma", param_range=param_range,
                cv=10, scoring="accuracy", n_jobs=1)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            plt.title("Validation Curve with SVM")
            plt.xlabel("$\gamma$")
            plt.ylabel("Score")
            plt.ylim(0.0, 1.1)
            plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2, color="r")
            plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                         color="g")
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2, color="g")
            plt.legend(loc="best")
            plt.show()
"""

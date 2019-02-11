# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:17:32 2018

@author: Chris McIntyre

Script is to analyze the Ames Real Estate dataset from Kaggle website.
"""
# Import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder


#load the data form CSV
DATA_PATH = os.path.join("datasets", "ames_data")

def load_data(filename, data_path=DATA_PATH):
    csv_path = os.path.join(data_path, filename)
    return pd.read_csv(csv_path)

train_data = load_data("train.csv")
test_data = load_data("test.csv")   #   Used for final submission

def preprocess_mixed_data(train_data, imp_strat, fill_val, y_label):                   
    #   Separate the object dtype from numbers
    X_train = train_data.select_dtypes(exclude=['object'])
    X_train_obj = train_data.select_dtypes(include=['object'])
    
    #   Remove NaN/missing numerical data
        #take a quick look at the data
    #sample_incomplete_rows = X_train[X_train.isnull().any(axis=1)].head()
    
    imputer = Imputer(strategy=imp_strat) #  Strategies are mean, median and most_frequent
    X = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(X, columns=X_train.columns) #convert back to DataFrame
    #   Look at the values that will be placed in each feature column?
    #stats = imputer.statistics_
    ##   Transform the data (apply the imputer to the original data)
    #X_train_cp = imputer.transform(X_train_cp)
    ##   Verify no incomplete data exists; should choke
    ##post_incomplete_rows = X_train_cp[X_train_cp.isnull().any(axis=1)].head()
    
    #   Work on the object types to encode categorical data and address missing data
    #   Mising Data; replace with string 'ZZ'
    X_train_obj = X_train_obj.fillna(fill_val)
    
    le = LabelEncoder()
    for obj in X_train_obj:
        X_train_obj[obj] = le.fit_transform(X_train_obj[obj])
    
    #   Join numerical and catergorical dataframes
    X_train = X_train.join(X_train_obj)
    if y_label:
        y_train = X_train[y_label]
        X_train = X_train.drop(y_label, axis=1)
    
    return y_train, X_train

y_train, X_train = preprocess_mixed_data(train_data, "median", "ZZ", "SalePrice")

#   Looks for correlations in the data
"""attributes = train_data.keys()
pd.scatter_matrix(train_data[attributes[65:80]], figsize=(20,12))
plt.show()
""" 
   
"""
Alorithms to try 
ensemble:
    AdaBoostRegressor;
    BaggingRegressor;
    ExtraTreesRegressor;
    GradientBoostingRegressor;
    RandomForestRegressor;
neighbors:
    KNeighborsRegressor;
    RadiusNeighborsRegressor;
neural_network:
    MLPRegressor;
svm:
    LinearSVR;
    NuSVR;
    SVR;
tree:
    DecisionTreeRegressor;
    ExtraTreeRegressor;


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
        
        EXAMPLE PLOTTING:
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
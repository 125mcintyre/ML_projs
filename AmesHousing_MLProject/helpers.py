# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:03:13 2018

@author: HMTV4826

Contain numerour useful helper functions for ML processing.

FUNCTIONS:
    preprocess_mixed_data(train_data, imp_strat, fill_val, y_label)
    load_data(filename, DATA_PATH)

"""

def load_data(filename, DATA_PATH):
    import os
    import pandas as pd 
    csv_path = os.path.join(DATA_PATH, filename)
    return pd.read_csv(csv_path)

def preprocess_mixed_data(train_data, imp_strat, fill_val, y_label, strat_scaler): 
    import pandas as pd                  
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    
    #   Separate the object dtype from numbers
    X_train = train_data.select_dtypes(exclude=['object'])
    X_train_obj = train_data.select_dtypes(include=['object'])
    
    #   Remove NaN/missing valus on numerical data    
    #  Strategies are mean, median and most_frequent
    imputer = Imputer(strategy=imp_strat) 
    X = imputer.fit_transform(X_train)
    #convert back to DataFrame
    X_train = pd.DataFrame(X, columns=X_train.columns) 
    #   Work on the object types to encode categorical data and address 
    #   missing data; replace with string fill_val
    X_train_obj = X_train_obj.fillna(fill_val)
    #   Encode all the categorical data
    le = LabelEncoder()
    for obj in X_train_obj:
        X_train_obj[obj] = le.fit_transform(X_train_obj[obj])
    #   Join numerical and catergorical DataFrames
    X_train = X_train.join(X_train_obj)
    
    if y_label == "":
        if strat_scaler:
            scaler = StandardScaler()
            X_scaler = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_scaler, columns=X_train.columns)
        return X_train
    else:
        y_train = X_train[y_label]
        X_train = X_train.drop(y_label, axis=1)
        if strat_scaler:
            scaler = StandardScaler()
            X_scaler = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_scaler, columns=X_train.columns)
        return y_train, X_train

def plot_learn_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])):
    import matplotlib.pyplot as plt
    from sklearn.learning_curve import learning_curve
    import numpy as np    

    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    
    return plt

def get_regression_estimator(item):

    from sklearn import ensemble
    from sklearn import svm 
    from sklearn import neural_network 
    from sklearn import neighbors
    from sklearn import tree

    if item == "RandomForestRegressor":
        param_grid = {"max_depth": [None],
                      "max_features": [30,35,40],
                      "bootstrap": [False, True],
                      "warm_start": [True, False]}
        estimator = ensemble.RandomForestRegressor(random_state=42)
        """
        n_estimators : integer, optional (default=10)
            The number of trees in the forest.
        criterion : string, optional (default=”mse”)
            The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.
            New in version 0.18: Mean Absolute Error (MAE) criterion.
        max_features : int, float, string or None, optional (default=”auto”)
            The number of features to consider when looking for the best split:
            •If int, then consider max_features features at each split.
            •If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
            •If “auto”, then max_features=n_features.
            •If “sqrt”, then max_features=sqrt(n_features).
            •If “log2”, then max_features=log2(n_features).
            •If None, then max_features=n_features.
             Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
        max_depth : integer or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:
            •If int, then consider min_samples_split as the minimum number.
            •If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
            Changed in version 0.18: Added float values for percentages.
        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node:
            •If int, then consider min_samples_leaf as the minimum number.
            •If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
            Changed in version 0.18: Added float values for percentages.
       min_weight_fraction_leaf : float, optional (default=0.)
            The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
       max_leaf_nodes : int or None, optional (default=None)
           Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
       min_impurity_split : float,
             Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
             Deprecated since version 0.19: min_impurity_split has been deprecated in favor of min_impurity_decrease in 0.19 and will be removed in 0.21. Use min_impurity_decrease instead.
       min_impurity_decrease : float, optional (default=0.)
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            The weighted impurity decrease equation is the following:
                    N_t / N * (impurity - N_t_R / N_t * right_impurity- N_t_L / N_t * left_impurity)
                        where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.
                        N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
                    New in version 0.19.
        bootstrap : boolean, optional (default=True)
            Whether bootstrap samples are used when building trees.
            oob_score : bool, optional (default=False)
            whether to use out-of-bag samples to estimate the R^2 on unseen data.
        n_jobs : integer, optional (default=1)
            The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.
            random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
        verbose : int, optional (default=0)
            Controls the verbosity of the tree building process.
        warm_start : bool, optional (default=False)
            When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest
        """
        return param_grid, estimator
    elif item == "AdaBoostRegressor":
        param_grid = {"n_estimators": [20, 30, 50, 75],
                      "learning_rate": [.5, .75, 1],
                      "loss": ["linear", "square", "exponential"]}
        estimator = ensemble.AdaBoostRegressor(random_state=42)
        """
        base_estimator : object, optional (default=DecisionTreeRegressor)
            The base estimator from which the boosted ensemble is built. Support for sample weighting is required.
        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
        learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each regressor by learning_rate. There is a trade-off between learning_rate and n_estimators.
        loss : {‘linear’, ‘square’, ‘exponential’}, optional (default=’linear’)
            The loss function to use when updating the weights after each boosting iteration.
        random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
        """
        return param_grid, estimator
    elif item == "BaggingRegressor":
        param_grid = {"max_features": [30, 35, 40],
                      "bootstrap": [False, True],
                      "bootstrap_features": [True, False],
                      "warm_start": [True, False]}
        estimator = ensemble.BaggingRegressor(random_state=42)
        """
            base_estimator : object or None, optional (default=None)
                The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.
            n_estimators : int, optional (default=10)
                The number of base estimators in the ensemble.
            max_samples : int or float, optional (default=1.0)
                The number of samples to draw from X to train each base estimator.
                •If int, then draw max_samples samples.
                •If float, then draw max_samples * X.shape[0] samples.
            max_features : int or float, optional (default=1.0)
                The number of features to draw from X to train each base estimator.
                •If int, then draw max_features features.
                •If float, then draw max_features * X.shape[1] features.
            bootstrap : boolean, optional (default=True)
                Whether samples are drawn with replacement.
            bootstrap_features : boolean, optional (default=False)
                Whether features are drawn with replacement.
            oob_score : bool
                Whether to use out-of-bag samples to estimate the generalization error.
            warm_start : bool, optional (default=False)
                When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble.
            n_jobs : int, optional (default=1)
                The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
            verbose : int, optional (default=0)
                Controls the verbosity of the building process.
        """
        return param_grid, estimator
    elif item == "ExtraTreesRegressor":
        param_grid = {"n_estimators": [15, 30, 45],
                      "max_depth": [None],
                      "max_features": ["auto"],
                      "min_samples_split": [2, 4, 10],
                      "min_samples_leaf": [1,5],
                      "bootstrap": [False, True],
                      "warm_start": [True, False]}
        estimator = ensemble.ExtraTreesRegressor(random_state=42)
        """
        n_estimators : integer, optional (default=10)
        criterion : string, optional (default=”mse”)
        max_features : int, float, string or None, optional (default=”auto”)
        max_depth : integer or None, optional (default=None)
        min_samples_split : int, float, optional (default=2)
        min_samples_leaf : int, float, optional (default=1)
        min_weight_fraction_leaf : float, optional (default=0.)
        max_leaf_nodes : int or None, optional (default=None)
        min_impurity_split : float,
        min_impurity_decrease : float, optional (default=0.)
        bootstrap : boolean, optional (default=False)
        oob_score : bool, optional (default=False)
        n_jobs : integer, optional (default=1)
        random_state : int, RandomState instance or None, optional (default=None)
        verbose : int, optional (default=0)
        warm_start : bool, optional (default=False)
        """
        return param_grid, estimator
    elif item == "GradientBoostingRegressor":
        param_grid = {"max_depth": [None],
                      "max_features": [30,35,45],
                      "loss": ["ls", "lad", "huber", "quantile"],
                      "alpha": [.9, .5],
                      "warm_start": [True, False]}
        estimator = ensemble.GradientBoostingRegressor(random_state=42)
        """
        loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)
            loss function to be optimized. ‘ls’ refers to least squares regression. 
            ‘lad’ (least absolute deviation) is a highly robust loss function solely 
            based on order information of the input variables. ‘huber’ is a 
            combination of the two. ‘quantile’ allows quantile regression 
            (use alpha to specify the quantile).
        learning_rate : float, optional (default=0.1)
        n_estimators : int (default=100)
            The number of boosting stages to perform. Gradient boosting is fairly 
            robust to over-fitting so a large number usually results in better 
            performance.
        max_depth : integer, optional (default=3)
            maximum depth of the individual regression estimators. The maximum 
            depth limits the number of nodes in the tree. Tune this parameter 
            for best performance; the best value depends on the interaction of 
            the input variables.
        criterion : string, optional (default=”friedman_mse”)
            The function to measure the quality of a split. Supported criteria are 
            “friedman_mse” for the mean squared error with improvement score by Friedman, 
            “mse” for mean squared error, 
            and “mae” for the mean absolute error. 
            The default value of “friedman_mse” is generally the best as it can 
            provide a better approximation in some cases.
        min_samples_split : int, float, optional (default=2)
        min_samples_leaf : int, float, optional (default=1)
        min_weight_fraction_leaf : float, optional (default=0.)
        subsample : float, optional (default=1.0)
            The fraction of samples to be used for fitting the individual base 
            learners. If smaller than 1.0 this results in Stochastic Gradient 
            Boosting. subsample interacts with the parameter n_estimators. Choosing 
            subsample < 1.0 leads to a reduction of variance and an increase in bias.
        max_features : int, float, string or None, optional (default=None)
        max_leaf_nodes : int or None, optional (default=None)
        min_impurity_split : float,
        min_impurity_decrease : float, optional (default=0.)
        alpha : float (default=0.9)
            The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'.
        init : BaseEstimator, None, optional (default=None)
            An estimator object that is used to compute the initial predictions. 
            init has to provide fit and predict. 
            If None it uses loss.init_estimator.
        verbose : int, default: 0
        warm_start : bool, default: False
        random_state : int, RandomState instance or None, optional (default=None)
        presort : bool or ‘auto’, optional (default=’auto’)
            Whether to presort the data to speed up the finding of best splits in fitting. Auto mode by default will use presorting on dense data and default to normal sorting on sparse data. Setting presort to true on sparse data will raise an error.
            New in version 0.17: optional parameter presort.
        """
        return param_grid, estimator
    elif item == "KNeighborsRegressor":
        param_grid = {}
        estimator = neighbors.KNeighborsRegressor()
        """
        n_neighbors : int, optional (default = 5)
            Number of neighbors to use by default for kneighbors queries.
        weights : str or callable
            weight function used in prediction. Possible values:
                •‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                •‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
                •[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
            Uniform weights are used by default.
        algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
                    Algorithm used to compute the nearest neighbors:
                        •‘ball_tree’ will use BallTree
                        •‘kd_tree’ will use KDTree
                        •‘brute’ will use a brute-force search.
                        •‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
                Note: fitting on sparse input will override the setting of this parameter, using brute force.
        leaf_size : int, optional (default = 30)
            Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
        p : integer, optional (default = 2)
            Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        metric : string or callable, default ‘minkowski’
            the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric class for a list of available metrics.
        metric_params : dict, optional (default = None)
            Additional keyword arguments for the metric function.
        n_jobs : int, optional (default = 1)
        """
        return param_grid, estimator
    elif item == "RadiusNeighborsRegressor":
        param_grid = {}
        estimator = neighbors.RadiusNeighborsRegressor()
        """
        radius : float, optional (default = 1.0)
                Range of parameter space to use by default for radius_neighbors queries.
        weights : str or callable
        algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
        leaf_size : int, optional (default = 30)
        p : integer, optional (default = 2)
        metric : string or callable, default ‘minkowski’
        metric_params : dict, optional (default = None)
        """       
        return param_grid, estimator
    elif item == "MLPRegressor":
        param_grid = {"activation": ["logistic", "relu"],
                      "solver": ["lbfgs", "adam"],
                      "learning_rate": ["constant", "invscaling", "adaptive"]}
        estimator = neural_network.MLPRegressor(random_state=42)
        """
        hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
            The ith element represents the number of neurons in the ith hidden layer.
        activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
             Activation function for the hidden layer.
                 •‘identity’, no-op activation, useful to implement linear bottleneck, 
                 returns f(x) = x
                 •‘logistic’, the logistic sigmoid function, returns 
                 f(x) = 1 / (1 + exp(-x)).
                 •‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
                 •‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
        solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
            The solver for weight optimization.
                •‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
                •‘sgd’ refers to stochastic gradient descent.
                •‘adam’ refers to a stochastic gradient-based optimizer proposed 
                by Kingma, Diederik, and Jimmy Ba
            Note: The default solver ‘adam’ works pretty well on relatively 
            large datasets (with thousands of training samples or more) in terms 
            of both training time and validation score. For small datasets, however, 
            ‘lbfgs’ can converge faster and perform better.
        alpha : float, optional, default 0.0001
            L2 penalty (regularization term) parameter.
        batch_size : int, optional, default ‘auto’
            Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, 
            the classifier will not use minibatch. When set to “auto”, 
            batch_size=min(200, n_samples)
        learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
            Learning rate schedule for weight updates.
                •‘constant’ is a constant learning rate given by ‘learning_rate_init’.
                •‘invscaling’ gradually decreases the learning rate learning_rate_ 
                at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. 
                effective_learning_rate = learning_rate_init / pow(t, power_t)
                •‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ 
                as long as training loss keeps decreasing. Each time two consecutive 
                epochs fail to decrease training loss by at least tol, or fail to 
                increase validation score by at least tol if ‘early_stopping’ is 
                on, the current learning rate is divided by 5.
            Only used when solver=’sgd’.
        learning_rate_init : double, optional, default 0.001
            The initial learning rate used. It controls the step-size in updating 
            the weights. Only used when solver=’sgd’ or ‘adam’.
        power_t : double, optional, default 0.5
            The exponent for inverse scaling learning rate. It is used in updating 
            effective learning rate when the learning_rate is set to ‘invscaling’. 
            Only used when solver=’sgd’.
        max_iter : int, optional, default 200
            Maximum number of iterations. The solver iterates until convergence 
            (determined by ‘tol’) or this number of iterations. For stochastic 
            solvers (‘sgd’, ‘adam’), note that this determines the number of epochs 
            (how many times each data point will be used), not the number of 
            gradient steps.
        shuffle : bool, optional, default True
            Whether to shuffle samples in each iteration. Only used when 
            solver=’sgd’ or ‘adam’.
        random_state : int, RandomState instance or None, optional, default None
        tol : float, optional, default 1e-4
            Tolerance for the optimization. When the loss or score is not improving 
            by at least tol for two consecutive iterations, unless learning_rate is 
            set to ‘adaptive’, convergence is considered to be reached and training 
            stops.
        verbose : bool, optional, default False
        warm_start : bool, optional, default False
            When set to True, reuse the solution of the previous call to fit as 
            initialization, otherwise, just erase the previous solution.
        momentum : float, default 0.9
            Momentum for gradient descent update. Should be between 0 and 1. 
            Only used when solver=’sgd’.
        nesterovs_momentum : boolean, default True
            Whether to use Nesterov’s momentum. Only used when solver=’sgd’ 
            and momentum > 0.
        early_stopping : bool, default False
            Whether to use early stopping to terminate training when validation 
            score is not improving. If set to true, it will automatically set 
            aside 10% of training data as validation and terminate training when 
            validation score is not improving by at least tol for two consecutive 
            epochs. Only effective when solver=’sgd’ or ‘adam’
        validation_fraction : float, optional, default 0.1
            The proportion of training data to set aside as validation set for 
            early stopping. Must be between 0 and 1. Only used if early_stopping 
            is True
        beta_1 : float, optional, default 0.9
            Exponential decay rate for estimates of first moment vector in adam, 
            should be in [0, 1). Only used when solver=’adam’
        beta_2 : float, optional, default 0.999
            Exponential decay rate for estimates of second moment vector in adam, 
            should be in [0, 1). Only used when solver=’adam’
        epsilon : float, optional, default 1e-8
            Value for numerical stability in adam. Only used when solver=’adam’
        """
        return param_grid, estimator
    elif item == "LinearSVR":
        param_grid = {"fit_intercept": [True, False],
                      #"dual": [True, False],
                      "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                      "epsilon": [0.1, 0]
                      }
        estimator = svm.LinearSVR(random_state=42)
        """
        C : float, optional (default=1.0)
            Penalty parameter C of the error term. The penalty is a squared l2 
            penalty. The bigger this parameter, the less regularization is used.
        loss : string, ‘epsilon_insensitive’ or ‘squared_epsilon_insensitive’ 
        (default=’epsilon_insensitive’)
            Specifies the loss function. ‘l1’ is the epsilon-insensitive loss 
            (standard SVR) while ‘l2’ is the squared epsilon-insensitive loss.
        epsilon : float, optional (default=0.1)
            Epsilon parameter in the epsilon-insensitive loss function. Note 
            that the value of this parameter depends on the scale of the target 
            variable y. If unsure, set epsilon=0.
        dual : bool, (default=True)
            Select the algorithm to either solve the dual or primal optimization 
            problem. Prefer dual=False when n_samples > n_features.
        tol : float, optional (default=1e-4)
        fit_intercept : boolean, optional (default=True)
            Whether to calculate the intercept for this model. If set to false,
            no intercept will be used in calculations (i.e. data is expected to 
            be already centered).
        intercept_scaling : float, optional (default=1)
            When self.fit_intercept is True, instance vector x becomes 
            [x, self.intercept_scaling], i.e. a “synthetic” feature with constant 
            value equals to intercept_scaling is appended to the instance vector. 
            The intercept becomes intercept_scaling * synthetic feature weight Note! 
            the synthetic feature weight is subject to l1/l2 regularization as all 
            other features. To lessen the effect of regularization on synthetic 
            feature weight (and therefore on the intercept) intercept_scaling 
            has to be increased.
        verbose : int, (default=0)
        random_state : int, RandomState instance or None, optional (default=None)
        max_iter : int, (default=1000)
        """
        return param_grid, estimator
    elif item == "NuSVR":
        param_grid = {"kernel": ["poly", "rbf", "sigmoid"],
                      "degree": [3,5,7,9]}
        estimator = svm.NuSVR()
        """
        C : float, optional (default=1.0)
        nu : float, optional
            An upper bound on the fraction of training errors and a lower bound 
            of the fraction of support vectors. Should be in the interval (0, 1]. 
            By default 0.5 will be taken.
        kernel : string, optional (default=’rbf’)
            Specifies the kernel type to be used in the algorithm. It must be 
            one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
            If none is given, ‘rbf’ will be used. If a callable is given it is used 
            to precompute the kernel matrix.
        degree : int, optional (default=3)
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        gamma : float, optional (default=’auto’)
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
        coef0 : float, optional (default=0.0)
            Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.
        tol : float, optional (default=1e-3)
        cache_size : float, optional
            Specify the size of the kernel cache (in MB).
        verbose : bool, default: False
        max_iter : int, optional (default=-1)
        """
        return param_grid, estimator
    elif item == "SVR":
        param_grid = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                      "degree": [3,5,7,9]}
        estimator = svm.SVR()
        """
        C : float, optional (default=1.0)
        epsilon : float, optional (default=0.1)
        kernel : string, optional (default=’rbf’)
        degree : int, optional (default=3)
        gamma : float, optional (default=’auto’)
        coef0 : float, optional (default=0.0)
        shrinking : boolean, optional (default=True)
        tol : float, optional (default=1e-3)
        cache_size : float, optional
        verbose : bool, default: False
        max_iter : int, optional (default=-1)
        """
        return param_grid, estimator
    elif item == "DecisionTreeRegressor":
        param_grid = {}
        estimator = tree.DecisionTreeRegressor(random_state=42)
        """
        DecisionTreeRegressor(criterion=’mse’, splitter=’best’, max_depth=None, 
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        max_features=None, random_state=None, max_leaf_nodes=None, 
        min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
        """
        return param_grid, estimator
    elif item == "ExtraTreeRegressor":
        param_grid = {}
        estimator = tree.ExtraTreeRegressor(random_state=42)
        """
        ExtraTreeRegressor(criterion=’mse’, splitter=’random’, max_depth=None, 
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        max_features=’auto’, random_state=None, min_impurity_decrease=0.0, 
        min_impurity_split=None, max_leaf_nodes=None)
        """
        return param_grid, estimator
    print("Estimator not in list.")
    return None, None

def plot_feature_importances(clf):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # #############################################################################
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
def plot_deviance(clf):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # #############################################################################
    # Plot training deviance
    
    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')


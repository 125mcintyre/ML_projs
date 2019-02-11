import pandas as pd


##  Load housing data
housing = pd.read_csv("Y:\Blk1A\Training\Machine Learning\Hands on ML With Python and Tensorflow\datasets\housing\housing.csv")

#Get first look at data
housing.head()
    #Notice how ocean proximity is th same value...indicating it is a category not unique values

#Get dataFrame information
housing.info()

#Look at the ocean_proximity value counts
housing["ocean_proximity"].value_counts()
    #This  will show you how manydistricts belog to each category

#use the describe() method to see other imporant information on all variables
housing.describe()

#plot a histogram to see how your data is distributed
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(15,10))

    #Look for oddities in your data like data caps-->see housing_median_age and median_house_value
    #Look for tail heavy distrobutions; these will need to be transformed later

#Caputre your test data bofore you look any further at the data.  You
#may accidentally overfit the data if you look to long at it.

# create a test set
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indicies = shuffled_indicies[:test_set_size]
    train_indicies = shuffled_indicies[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies] #return the index locations of training and test sets

#How to use
train_set, test_set = split_train_test(housing, 0.2) # 20% test_set
print(len(train_set), "train + ", len(test_set), "Test")

########    This works pretty well for the first time through
########    To get consistant results seed the random method "np.random.seed(42)".
########    Also useful to split the labels dataset identically.
########    See pg 50 of HO Machine Learning book for another implementation
#################################################################################################
## Here is a sklearn built-in function
from sklearn.model_selection import train_test_split
#Does the same as split_train_test above
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#   The next important thing to remember is to try and not add apmle bias to your data
#   Meaning ensure by haveing a certain number of samples of a particular type you could
#   cause the data to be biased toward the dominant samples.

#   Here is an example of how to divide a sample feature such that you have an even
#   distribution of strata.  Note: there should not be too many strata but not too large.

#   First break up the sample category into strata
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] <5, 5.0, inplace=True)

housing["income_cat"].hist()


#   Now lets use sklearns StratifiedShuffleSplit class

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)
#   Remove income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

##################
#   Moving on to Visualizing the data
#   Good practice to just make a copy; consider removing some of the data
#   to make the data easier on your computer.
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")

#   More granularity
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

#   Adding additional details
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
#plt.show()

#   Check for correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
corr_matrix["median_income"].sort_values(ascending=False)
#   etc.
#############
#visualize the correlation using Pandas scatter_matrix
#from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income","total_rooms","housing_median_age","latitude","households"]
pd.scatter_matrix(housing[attributes], figsize=(12,8))
#plt.show()

#   You will notice a few positive correlations in the plots above.  You can also see there are
#   a few areas in the data that need to be cleaned up. The $500K data and 450, 350 280 have vertical lines.
#   Something is going on in the data set there and they may want to be removed else the ML algorithm will learn
#   these quirks. (HOML)

#   HOML recommends computing the logrithms of the tail heavy data as an option to get a better gausian distribution.

#   CREATING new attributes from existing ones.
housing["rooms_per_houshold"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["households"]/housing["population"]
#   Look at the correlation again
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
    #   you can see markedly better correlation in the data now
attributes = ["median_house_value", "median_income", "bedrooms_per_room","rooms_per_houshold", "population_per_household"]
pd.scatter_matrix(housing[attributes], figsize=(12,8))
#plt.show()
    #   Can't see much here
#####################
#   Data Preprocessing
housing = strat_train_set.drop("median_house_value", axis=1)
    # creates a copy of strat_train_set (your training set) and drops the labels
housing_labels = strat_train_set["median_house_value"].copy()
    #   creates the labels

#   Data cleaning
#   handling missing data

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

  #   Three options 
    #   1) drop anything that is not a number
housing.dropna(subset=["total_bedrooms"])
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
    #   2) remove the entire feature
housing.drop("total_bedrooms",axis=1)
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
    #   3) replace the missing value with zero, mean, median, etc
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
    #imputer needs to have all numbers in the data set....need to remove the
    #ocean_proximity column
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
    #   Calculated the medians
imputer.statistics_
#   Now we can use the imputer to transform the housing_num data
X = imputer.transform(housing_num)
#   Convert it back to a pandas data frame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#   Now, what to do about the ocean_proximity feature.  It's categorical text data...
housing_cat = housing["ocean_proximity"]
housing_cat.head(10)    #   looking at the first ten instances
#   Use the factorize() method to generate a numberical dataset that can be mapped back
#   to the categories via the housing_categories return variable
housing_cat_encoded, housing_categories = housing_cat.factorize()
print(housing_categories)
print(housing_cat_encoded[:10])

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot

#   The OneHotEncoder returns a sparse array by default, but we can convert it to a dense array if needed:
    #housing_cat_1hot.toarray()
##################################
#   Code added because CategoricalEncoder not yet available
##################################
# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

#############################
#############################
#from sklearn.preprocessing import CategoricalEncoder # in future versions of Scikit-Learn

#cat_encoder = CategoricalEncoder()
#housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
#housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
#housing_cat_1hot

##########
#   Lets create a custom tranformer
##########
from sklearn.base import BaseEstimator, TransformerMixin

# column index based on dataset formatting
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):   #   no *args or **kargs gets
                                                        #   you the get and set functions
                                                        #   for free
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
#   How to use it
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    #   By adding this as a hyperparameter you can use it to auto tune hyperparameters
    #   later to optimize your algorithm.  You can add as many as you like if you
    #   do not yet know the best combination of features.
housing_extra_attribs = attr_adder.transform(housing.values)

##########
#   Feature scaling
#   1) Min-Max Scaling (sklearn MinMaxScaler or Normalization will change you feature values to values
#   from 0-1.  This is performed by subtracting the min value then dividing by the max-min.
#   2) Standardization (sklearn StandardScaler) is different.  First subtract the mean from every value so you have a mean of zero.
#   Then divide by the variance so the distribution has unit variance.

##########
#   Transformation Pipelines
#   Example
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', Imputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet 
#REPEATED: from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#   Now remember you needed to have already identified the categorical
#   and numerical data once.  We will use these lists to set our DataFrameSelector
#   class attribute_names.
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# Create a class to select numerical OR categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
        
#   Now lets create two pipelines to use our new DataFrameSelector, one for
#   Nums and one for Cats

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding = "onehot-dense"))
    ])
#   We can now combine all our pipelines in to one
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion([
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
    ])

#   Now to use it
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)
print("Preprocessing is working!!!!!!!!")

#####################
#   Training

#   Training a Linear Regression model and RMSE
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#   Pull some data out and check its performance
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("predictions:", lin_reg.predict(some_data_prepared))
print("labels", list(some_labels))
    # Doesn't work super great
#   Lets calc RMSE
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
    #   You can see that the model is underfitting the data it's off by $68k
    #   Let's try another model
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
#   Check it out
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)
    #   "0" -->This is not possible....clearly overfitting the data
#   Let's try Cross-Validation to break up our training set into a traning and
#   validation test set.
#   k-folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                          scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
#   function to display scores
def display_scores(scores):
	print("scores: ", scores)
	print("Mean: ", scores.mean())
	print("Standard Deviation: ", scores.std())

display_scores(tree_rmse_scores)
#   Okay we went from underfitting with the lin and overfitting with the
#   tree.  We could clc the lin_reg cross validation scores as well if we wanted to.
#   Let's move on to another algorithm. RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
#   Training Performance
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
#   Validation Performance
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                          scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
#   training RMSE = 22289 and cv RMSE = 52964
#   So the algorithm is still overfitting the training data.  you might try a
#   max of 3-5 models to try to hone in on the best ones before messing with
#   hyperparameter optimization.
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
    #   Worse...
#   HOML recommends saving the models once you are complete using the following
#   pickle module
#from sklearn.externals import joblib

#jodlib.dump(my_model, "my_model.pkl")
#my_model_loaded = loblib.load("my_model.pkl")

######
#   Fine-Tuning your models
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap':[False], 'n_estimators': [3,10], 'max_features': [2,3,4]},
    ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
#   {'max_features': 8, 'n_estimators': 30}
#   You can also get the best estimator directly by grid_search.best_estimator_
##RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
##           max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
##           min_impurity_split=None, min_samples_leaf=1,
##           min_samples_split=2, min_weight_fraction_leaf=0.0,
##           n_estimators=30, n_jobs=1, oob_score=False, random_state=42,
##           verbose=0, warm_start=False)


#   Let's look at the score of each hyperparameter combination tested during
#   the grid search
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
####
#   Try a randomized Grid search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
#   Let's look at the scores for the hyperparameters tested for the 10 iterations with 5 folds cross-validation
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
##    49147.1524172 {'max_features': 7, 'n_estimators': 180}
##    51396.8768969 {'max_features': 5, 'n_estimators': 15}
##    50798.3025423 {'max_features': 3, 'n_estimators': 72}
##    50840.744514 {'max_features': 5, 'n_estimators': 21}
##    49276.1753033 {'max_features': 7, 'n_estimators': 122}
##    50776.7360494 {'max_features': 3, 'n_estimators': 75}
##    50682.7075546 {'max_features': 3, 'n_estimators': 88}
##    49612.1525305 {'max_features': 5, 'n_estimators': 100}
##    50472.6107336 {'max_features': 3, 'n_estimators': 150}
##    64458.2538503 {'max_features': 5, 'n_estimators': 2}


####################
#   Let's wrap it up and implement our pipeline on our test set and see how we performed!

#   Okay we did our best....lets save the best estimator model as our final_model
final_model = rnd_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"]

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)




### Turn Plots on at the end...?
#plt.show()


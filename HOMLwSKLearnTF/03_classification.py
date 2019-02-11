from load_mat_dataset import *
import numpy as np

#load our MNIST-orginal dataset (currently a .mat file)
path = "./datasets/mnist/mnist-original.mat"
mnist = import_dataset(path, "MNIST-original")

X , y = mnist["data"],mnist["target"]
print("X shape: ", X.shape)
print("y shape: ", y.shape)

#   Let's take a look at one of the pictures.  Need to reshape since MNIST data is in column format.
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[35000]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
#   Print the label
print("Pictures label: ", y[35000])

#   Want to plot a group of digits?
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

def plot_digit(instance, **options):
    size = 28
    image = instance.reshape(size,size)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
    
    
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
#save_fig("more_digits_plot")

#   so that all works fine...
#########
#   Create the test and training sets.  Then shuffle the trianing set to get 
#   all the numbers and none that are in a row.
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#   Needs: import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#########
#   Classifier: Binary
#       Model implementation: Stocastic Gradient Descent (SGD)
#       Goal: Determine if is a 5 or not

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(n_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)

print("Is a 5 prediction: ", sgd_clf.predict(some_digit))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("Accuracy scores: ", scores)
#   This is pretty good---> but only because it has so many not 5 values
#   if it were to guess everything as not a 5 it would be correct 90% of the time.
#   This is not a good measure of the models perfomance.  Let's look at the confusion matrix
#   for this model
from sklearn.model_selection import cross_val_predict

#   Let's use cross_val_predict, which performs K-folds cross validation but does not
#   return the evaluation scores but returns the predictions made on each fold.  you get a clean
#   prediction for each instance in the training set.
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#   now load up the confusion matrix
from sklearn.metrics import confusion_matrix

print("Confusion matrix: ", confusion_matrix(y_train_5, y_train_pred))
#   Other useful metrics are precision and recall
from sklearn.metrics import precision_score, recall_score

print("SGD Precision: ", precision_score(y_train_5, y_train_pred))
print("SGD Recall: ", recall_score(y_train_5, y_train_pred))

#   more useful is the F1 score
from sklearn.metrics import f1_score

print("F1 score: ", f1_score(y_train_5, y_train_pred))
"""
############### FYI     #############################
       nice way to make a training set quickly
    from sklearn.datasets import make_classification
    
    data, target = make_classification(n_samples=2500,
                                       n_features=45,
                                       n_informative=15,
                                       n_redundant=5)

ROC Curves

   Get all the "decision scores" similar to how we did the prediction values 
   above
"""
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr, "SGD")
#plt.show()

#   Calculating the Area Under the Curve of the ROC is simple
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_train_5, y_scores)
print("SGD AUC: ", auc)

######
#   Now train a random forest classifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_proba_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')

y_scores_forest = y_proba_forest[:,1]    #   score = probab of positive class
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

#   Plot both curves SGD and RF
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
#   Calculate AUC
auc_forest = roc_auc_score(y_train_5, y_scores_forest)
print("Forest AUC: ", auc_forest)
#   Calculate Precision and Recall
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
print("Forest Precision: ", precision_score(y_train_5, y_train_pred_forest))
print("Forest Recall: ", recall_score(y_train_5, y_train_pred_forest))

#plt.show()
"""
#####################
   Moving on to multiclass classification

   One way to create a mulitclass classifier is to use multiple binary 
   classifiers to perform multiple individual decisions. Looking at all those decisions 
   you can see which one has the highest score and that classifiers index maps 
   to the correct label.  For instance the "is 5 classifier" is activated.
   Here it is in action:
"""
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

some_digit_scores = sgd_clf.decision_function([some_digit])
clf_max = np.argmax(some_digit_scores)
print("SGD Prediction==Label?: ", (sgd_clf.predict([some_digit]) == sgd_clf.classes_[clf_max]), " -->Success!!!")

"""
   This looks like it did a good job.  The default use for using a binary 
   classifier as a mulitclass classifier is the One-vs-All (OvA) method, except 
   SVC that use One-vs-One(OvO). You can force either function with 
   OnevsOneClassifier(SGDClassifier(random_state=42)) or
   OnevsRestClassifier(SGDClassifier(random_state=42))
"""
#   Random Forest implementation
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

some_digit_scores = forest_clf.predict_proba([some_digit])
clf_max = np.argmax(some_digit_scores)
print("Forest Prediction==Label?: ", (forest_clf.predict([some_digit]) == forest_clf.classes_[clf_max]), " -->Success!!!")


#   Getting back to validating performance, look at the CV scores
print("Unscaled SGD Cross Validations Scores: ", cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

#   This is good but here's a little trick!!
#   Remember that scaling of input data is always a good idea (CH2 HOML)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print("Scaled SGD Cross Validation Scores: ", cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

#   Lastly perform error analysis.  Where is the classifier having trouble?
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print("Confusion maxrix: ", conf_mx)

      # Plot it using matplotlib
def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

plt.matshow(conf_mx, cmap=plt.cm.gray)

#plt.show()

"""
    Focusing on the errors; plot data such that the brightest cells contain the 
   error.  You will see that 3's and 5's, understandable, are confused.  The
   rows in this plot are the actual classes and the columns are the predited
   classes.  So notice how the columns for 8&9 are brighter...this means that
   8's & 9's are commonly misclassified.  Also, looking at rows 8&9 you can see
   that they are commonly classified incorretly as some other number.
   
   Could engineer new features for 8's and 9's if you want to.  e.g. look for
   the number of loops in an image.
"""
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

#   Take a look at the 3's and 5's 
      
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
#plt.show()

"""
    MulitLabel Classification
    This is where there ould be multiple labels givent to each instance.
    Here is an implementation of one approach for this.  
    BTW: not all classifiers support multilabel calssification.
"""

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)  #   is large number?
y_train_odd = (y_train % 2 == 1)    # is odd?
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

#   For performance metrics on mulit label...
#   This could take a little while...
#y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)   # n_jobs is how many cores you want to use
#score = f1_score(y_multilabel, y_train_knn_pred, average="macro")
#print("F1 Score - macro: ", score) 
#    # change average="weighted" to allow for weighting for more common 
#    # occurances in the dataset
#score = f1_score(y_multilabel, y_train_knn_pred, average="weighted")
#print("F1 Score - weighted: ", score)
"""
    Multioutput Classification
    Now we look at creating multiple output values for one output.
    This example outputs a value 0-255 for each output.  Each output is a pixel
    in an output image.  
    The goal of this is to remove noise from an image.
"""    
#   First add noise to the mnist dataset
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

#   take look at it
some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)

print("Done!!!")
#
#"""
#Now lets try to be smarter here and let grid search optimize our algorithm.
#
#"""
#from sklearn.model_selection import GridSearchCV
#
#param_grid = [{
#        'weights': ["uniform", "distance"], 
#        'n_neighbors': [3, 4],
#        }]
##   Note this will also take a while...on a 4 core about 23 hours I should think...
#knn2_clf = KNeighborsClassifier()
#grid_search = GridSearchCV(knn2_clf, param_grid, cv=2, verbose=3, n_jobs=-1)
#grid_search.fit(X_train, y_train)
#
#print("Best Parameters: ", grid_search.best_params_)
#print("Best Score: ", grid_search.best_score_)
#
#from sklearn.metrics import accuracy_score
#
#y_pred = grid_search.predict(X_test)
#print("Accuracy Score using Gridsearch: ", accuracy_score(y_test, y_pred))





plt.plot()
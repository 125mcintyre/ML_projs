# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:01:30 2018

@author: HMTV4826

Purpose: the intent is to see if I can get the survival prediction to the highest 
possible.  Not sure yet what the HOML code was able to do.  This will be a 
learning exercise on the fundamentals taught from chapters 1-3.

Let's see how well I do.  I will place thier code at the end and try to print 
the delta's.
"""

#load the data form CSV

import os
import pandas as pd

TITANIC_PATH = os.path.join("datasets", "titanic")

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)

train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")
test_targets = load_titanic_data("test_targets.csv")


    

"""
A little about the data:
    Survived: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
    Pclass: passenger class.
    Name, Sex, Age: self-explanatory
    SibSp: how many siblings & spouses of the passenger aboard the Titanic.
    Parch: how many children & parents of the passenger aboard the Titanic.
    Ticket: ticket id
    Fare: price paid (in pounds)
    Cabin: passenger's cabin number
    Embarked: where the passenger embarked the Titanic

    New:
        SibSpParCh:     how many have both siblings&spouses and children&parents
"""
#   Take a look at the rows and columns 
print(train_data.head())
"""
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S 

"""

#   Check for missing data
print(train_data.info())
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
"""

#   Look for numerical attributes
print(train_data.describe())
"""
PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  714.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
std     257.353842    0.486592    0.836071   14.526497    1.102743   
min       1.000000    0.000000    1.000000    0.420000    0.000000   
25%     223.500000    0.000000    2.000000   20.125000    0.000000   
50%     446.000000    0.000000    3.000000   28.000000    0.000000   
75%     668.500000    1.000000    3.000000   38.000000    1.000000   
max     891.000000    1.000000    3.000000   80.000000    8.000000   

            Parch        Fare  
count  891.000000  891.000000  
mean     0.381594   32.204208  
std      0.806057   49.693429  
min      0.000000    0.000000  
25%      0.000000    7.910400  
50%      0.000000   14.454200  
75%      0.000000   31.000000  
max      6.000000  512.329200 

"""
##################################
### Start own analysis
#   Create the labels for train and test

"""
Since the question to answer is not who survived but will they survive, I don't
see how the persons name would matter or match any of the test sets names.  
Nor is there enough to train the model to recognize a particular name since 
there are not very many in each name group.

Also, Cabin is missing a ton of data.  This data will not be helpful in the 
final prediciton of surviving.  Though the information is captured in the 
fare cost which indirectly says what deck they would be on.

I will remove Name, Ticket and Cabin columns from the data.  Then I will need to 
convert the Sex and Embarked data to numerical data.
"""
y_train = train_data["Survived"]
X_train = train_data.drop("Survived", axis=1)
y_test = test_targets["Survived"]
X_test = test_data

for row in range(0,X_train.shape[0]):
    X_train.loc[row,"SibSpParCh"] = ((X_train.loc[row, "SibSp"] >= 1) and (X_train.loc[row, "Parch"] >= 1))
X_train["Older"] = X_train["Age"] > 55

print("\n\n\n")
print("Loaded SibSpParCh and Older attributes.\n")
print(X_train.head())











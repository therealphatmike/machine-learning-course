import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# read data set, determine features and dependent variable vector
data_set = pd.read_csv('data.csv')
features = data_set.iloc[:, :-1].values
dependent_vector = data_set.iloc[:, -1].values

# fill in missing data by averages
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(features[:, 1:3])
features[:, 1:3] = imputer.transform(features[:, 1:3])

# one hot encode categorical data so as not to force numerical meaning where there is none
# this will encode string categories into unit vectors so each is numerically independent
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
features = np.array(ct.fit_transform(features))

# label encode dependent variable so "yes" "no" values are encoded as 1s and 0s
le = LabelEncoder()
dependent_vector = le.fit_transform(dependent_vector)

# split data set into training data and testing data
feature_train, feature_test, dependent_train, dependent_test = train_test_split(
  features, dependent_vector, test_size=0.2, random_state=0
)

# feature scaling - using standardization scaling
# standard scaling is (x - mean(x))/standard deviation(x)
# fit_transform will fit the scaling and then transform the matrix
# then we call transform on the test set. We only call transform
# becuase part of this process of scaling is to use the same mean
# and standard deviation we calculated with the training set on the test and production data
sc = StandardScaler()
feature_train[:, 3:] = sc.fit_transform(feature_train[:, 3:])
feature_test[:, 3:] = sc.transform(feature_test[:, 3:])

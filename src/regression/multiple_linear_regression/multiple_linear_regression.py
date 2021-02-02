import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

data_set = pd.read_csv('src/regression/multiple_linear_regression/50_Startups.csv')
features = data_set.iloc[:, :-1].values
dependent_vector = data_set.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
features = np.array(ct.fit_transform(features))

feature_train, feature_test, dependent_train, dependent_test = train_test_split(
  features, dependent_vector, test_size=0.2, random_state=0
)

print("features training set: " + str(feature_train))
print("dependent var training set: " + str(dependent_train))
print("features test set: " + str(feature_test))
print("dependent var test set: " + str(dependent_test))
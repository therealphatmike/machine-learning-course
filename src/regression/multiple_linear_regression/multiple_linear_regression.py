import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

regressor = LinearRegression()
regressor.fit(feature_train, dependent_train)

test_prediction = regressor.predict(feature_test)
np.set_printoptions(precision=2)
reshaped_test_pred = test_prediction.reshape(len(test_prediction), 1)
reshaped_dependent_pred = dependent_test.reshape(len(dependent_test), 1)
concatenated_sets = np.concatenate((reshaped_test_pred, reshaped_dependent_pred), 1)
print("concatenated test prediction vs test acutal: " + str(concatenated_sets))
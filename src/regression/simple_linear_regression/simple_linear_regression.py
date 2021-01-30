import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data_set = pd.read_csv('src/regression/simple_linear_regression/salaries.csv')
features = data_set.iloc[:, :-1].values
dependent_vector = data_set.iloc[:, -1].values

feature_train, feature_test, dependent_train, dependent_test = train_test_split(
  features, dependent_vector, test_size=0.2, random_state=0
)

print(feature_train)
print(feature_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_set = pd.read_csv('src/regression/simple_linear_regression/salaries.csv')
features = data_set.iloc[:, :-1].values
dependent_vector = data_set.iloc[:, -1].values

feature_train, feature_test, dependent_train, dependent_test = train_test_split(
  features, dependent_vector, test_size=0.2, random_state=0
)

print('feature train: ' + str(feature_train))
print('feature test: ' + str(feature_test))

regressor = LinearRegression()
regressor.fit(feature_train, dependent_train)

train_prediction = regressor.predict(feature_train)
test_prediction = regressor.predict(feature_test)
print('train prediction: ' + str(train_prediction))
print('test prediction: ' + str(test_prediction))

plt.scatter(feature_train, dependent_train, color='red')
plt.plot(feature_train, train_prediction, color='blue')
plt.title('Salary vs Experience Training Set Results')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(feature_test, dependent_test, color='red')
# plotting the training line here, because the two are following the same
# linear regression model (the equation should be the same for both data sets)
# this will also provide a visual indicator of how close our training models the data
plt.plot(feature_train, train_prediction, color='blue')
plt.title('Salary vs Experience Test Set Results')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

# let's predict the salary of people with 12 years and 20 years of experience, respectively
print("Salary with 12 years of experience: "  + str(regressor.predict([[12]])))
print("Salary with 12 years of experience: "  + str(regressor.predict([[20]])))

# and finally, lets examine the model our training set created
print("Salary = " + str(regressor.coef_) + " x YearsExperience + " + str(regressor.intercept_))

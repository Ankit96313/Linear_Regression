# print(5 + (~2 +1))

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error 

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print(diabetes)

diabetes_x = diabetes.data[:,np.newaxis,2]
# diabetes_x = diabetes.data

# Split the data into training/testing sets

diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
# print(diabetes_y_train)
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_x_test)

print("error", mean_squared_error(diabetes_y_test,diabetes_y_predicted))

print("weighs",model.coef_)
print("intercept",model.intercept_)

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predicted)
plt.show()



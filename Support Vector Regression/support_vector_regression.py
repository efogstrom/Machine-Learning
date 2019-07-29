#SVR
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))

# Fitting the SVR to the dataset
# Create your regressor here
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('SVR Results')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the readable SVR results
plt.scatter(sc_y.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_y.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('SVR Results')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
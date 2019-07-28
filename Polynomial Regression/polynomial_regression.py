#Polynomial regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Fitting Linear Regression to the dataset ** Reference **

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
 

#Fitting Polynomial Regression to the dataset 

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing the linear regression results

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial regression results

plt.scatter(X, y, color = 'red')                            #orginal linear results
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')      #using real X results Predict Y
plt.title('Polynomial Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualizing more defined Polynomial Regression results

X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')                            #orginal linear results
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')      #using real X results Predict Y
plt.title('Polynomial Regression Results Optimized X Values')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with linear regression
print(lin_reg.predict([[6.5]]))


#Predicting a new result with polynomial regression

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
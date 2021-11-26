import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
dataset = pd.read_csv('Salary_Data.csv')
#print(dataset.shape)
#print(dataset.head())
dataset.plot(x='YearsExperience', y='Salary', style='o')
plt.title('YearsExperience vs Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
Linregressor = LinearRegression()
Linregressor.fit(X_train, y_train)
Logregressor = LogisticRegression(max_iter=1000)
Logregressor.fit(X_train, y_train)
Lin_pred = Linregressor.predict(X_test)
Log_pred = Logregressor.predict(X_test)
print("LinearRegression result")
print(pd.DataFrame({'Actual': y_test, 'Predicted': Lin_pred}))
print("--------------------------------------")
print("LogisticRegression result")
print(pd.DataFrame({'Actual': y_test, 'Predicted': Log_pred}))
print("--------------------------------------")
print("LinearRegression metrics")
print('MAE:', metrics.mean_absolute_error(y_test, Lin_pred))
print('MSE:', metrics.mean_squared_error(y_test, Lin_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Lin_pred)))
print("--------------------------------------")
print("LogisticRegression metrics")
print('MAE:', (metrics.mean_absolute_error(y_test, Log_pred)))
print('MSE:', metrics.mean_squared_error(y_test, Log_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Log_pred)))
print("--------------------------------------")
plt.scatter(x,y,s=10, color = 'b')
plt.plot([min(x), max(x)], [min(Lin_pred), max(Lin_pred)], color='y')
plt.show()

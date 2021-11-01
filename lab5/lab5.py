import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
raw_data = pd.read_csv('classified_data.csv', index_col = 0)
#print(raw_data.head())
scaler = StandardScaler()
scaler.fit(raw_data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(raw_data.drop('TARGET CLASS', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop('TARGET CLASS', axis=1).columns)
x = scaled_data
y = raw_data['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(x, y,train_size = 0.8, random_state = 1)
CART = DecisionTreeClassifier()
CART = CART.fit(X_train,y_train)
y_pred = CART.predict(X_test)
print(classification_report(y_test, y_pred))
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
plt.figure(figsize = (10, 10))
tree.plot_tree(CART, feature_names = X_train.columns, class_names= ['1', '0'], filled = True)
plt.show()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from c45 import C45
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
raw_data = load_breast_cancer()
clf = C45(attrNames=raw_data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(raw_data.data, raw_data.target, test_size=0.25)
c = clf.fit(X_train, y_train)
c45_pred = c.predict(X_test)
print("-----------------------------------------------------")
print("C4.5 result")
print("-----------------------------------------------------")
print(classification_report(y_test, c45_pred))
print("-----------------------------------------------------")
print("C4.5 Actual vs Predicted")
print("-----------------------------------------------------")
print(pd.DataFrame({'Actual': y_test, 'Predicted': c45_pred}))
print("-----------------------------------------------------")
CART = DecisionTreeClassifier()
CART = CART.fit(X_train,y_train)
CART_pred = CART.predict(X_test)
print("CART result")
print("-----------------------------------------------------")
print(classification_report(y_test, CART_pred))
print("-----------------------------------------------------")
print("CART Actual vs Predicted")
print("-----------------------------------------------------")
print(pd.DataFrame({'Actual': y_test, 'Predicted': CART_pred}))
print("-----------------------------------------------------")



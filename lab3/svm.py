import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
raw_data = pd.read_csv('classified_data.csv', index_col = 0)
#print(raw_data.head())
scaler = StandardScaler()
scaler.fit(raw_data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(raw_data.drop('TARGET CLASS', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop('TARGET CLASS', axis=1).columns)
x = scaled_data
y = raw_data['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(x, y,train_size = 0.1, random_state = 1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svc = SVC(C=1.0, random_state=1, kernel='linear')
svc.fit(X_train_std, y_train)
y_predict = svc.predict(X_test_std)
print(classification_report(y_test, y_predict))


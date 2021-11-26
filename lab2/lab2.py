import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

raw_data = pd.read_csv('Phishing_paper1.csv')
#print(raw_data)
print(raw_data.Status.value_counts())
to_remove1 = np.random.choice(raw_data[raw_data['Status']==0].index,size=450000,replace=False)
new = raw_data.drop(to_remove1)
print(new)
print(new.Status.value_counts())
scaler = StandardScaler()
scaler.fit(new.drop('Status', axis=1))
scaled_features = scaler.transform(new.drop('Status', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = new.drop('Status', axis=1).columns)
print(scaled_data)
x = scaled_data
y = new['Status']
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)
print(classification_report(y_test_data, predictions))

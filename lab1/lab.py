from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import urllib.request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
raw_data = pd.read_csv('Phishing_paper1.csv')
to_remove = np.random.choice(raw_data[raw_data['Status']==0].index,size=450000,replace=False)
new = raw_data.drop(to_remove)
scaler = MinMaxScaler()
scaler.fit(new.drop('Status', axis=1))
scaled_features = scaler.transform(new.drop('Status', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = new.drop('Status', axis=1).columns)
print(scaled_data)
features = scaled_data
labels = new['Status']
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size= .33, random_state = 14)
mlnNB = MultinomialNB() 
mlnNB.fit(features_train, labels_train)
pred_on_test_data = mlnNB.predict(features_test)
#print(pred_on_test_data)
predictions = mlnNB.predict(features_test)
print(classification_report(labels_test, predictions))
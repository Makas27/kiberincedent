from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import urllib.request

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url) 

dataset = np.loadtxt(raw_data, delimiter = ',')
print (dataset[0])
features = dataset[:, 0:48]
labels = dataset[:, -1]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size= .33, random_state = 14)
mlnNB = MultinomialNB() 
mlnNB.fit(features_train, labels_train)
pred_on_test_data = mlnNB.predict(features_test)
#print(pred_on_test_data)
score =  accuracy_score(pred_on_test_data, labels_test)
print ("Accuracry: ", score ,"%")
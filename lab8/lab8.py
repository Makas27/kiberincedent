from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, normalize
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA

raw_data = pd.read_csv('data.csv')
raw_data = raw_data.drop('CUST_ID', axis = 1)
raw_data.fillna(method ='ffill', inplace = True) 
scaler = StandardScaler() 
scaled_data = scaler.fit_transform(raw_data)
normalized_data = normalize(scaled_data)
normalized_data = pd.DataFrame(normalized_data)
 
plt.title('Clustering') 
Dendrogram = shc.dendrogram((shc.linkage(normalized_data, method ='ward')))
plt.show()
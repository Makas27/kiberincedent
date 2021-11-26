from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
data = pd.DataFrame(X)
std_slc = StandardScaler()
X_std = std_slc.fit_transform(X)
clt = DBSCAN()
model = clt.fit(X_std)
clusters = pd.DataFrame(model.fit_predict(X_std))
data["Cluster"] = clusters
fig = plt.figure(figsize=(10,10)); ax = fig.add_subplot(111)
scatter = ax.scatter(data[0],data[1], c=data["Cluster"],s=50)
ax.set_title("DBSCAN Clustering")
ax.set_xlabel("X0")
ax.set_ylabel("X1")
plt.colorbar(scatter)
#plt.show()
print(data[0], data[2])
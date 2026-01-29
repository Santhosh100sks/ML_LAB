import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski


df=pd.read_csv("Lab Session Data(IRCTC Stock Price).csv")
def replace_str_num(col):
      return col.astype(str).str.replace(",","").astype(float)

df["Price"] = replace_str_num(df["Price"])
df["Low"]   = replace_str_num(df["Low"])
df["Open"]  = replace_str_num(df["Open"])

X = df[["Open","Low"]].values[:249]
print(X)
y = (df["Price"].values[:249] > df["Open"].values[:249]).astype(int)
print(y)
# A2 : Intraclass spread & Interclass distance
def class_centroid(X,y,label):
    return np.mean(X[y==label],axis=0)

def class_spread(X,y,label):
    return np.std(X[y==label],axis=0)

def interclass_dist(c1,c2):
    return np.linalg.norm(c1-c2)

centroid0 = class_centroid(X,y,0)
centroid1 = class_centroid(X,y,1)

spread0 = class_spread(X,y,0)
spread1 = class_spread(X,y,1)

inter_dist = interclass_dist(centroid0,centroid1)

print("Centroid 0:",centroid0)
print("Spread  0:",spread0)
print("Centroid 1:",centroid1)
print("Spread  1:",spread1)
print("Interclass Distance:",inter_dist)

# a3 

open_feat = X[:,0]
mean_open = np.mean(open_feat)
var_open  = np.var(open_feat)

plt.hist(open_feat,bins=20)
plt.title("Histogram of Open Price")
plt.show()
print("Mean:",mean_open)
print("Variance:",var_open)
# A4 : Minkowski dis

def my_minkowski(a,b,p):
    return (np.sum(np.abs(a-b)**p))**(1/p)

v1 = X[0]
v2 = X[5]

p_list=[]
d_list=[]

for p in range(1,11):
    p_list.append(p)
    d_list.append(my_minkowski(v1,v2,p))

plt.plot(p_list,d_list,marker='o')
plt.xlabel("p")
plt.ylabel("Distance")
plt.title("Minkowski Distance vs p")
plt.show()

# A5 

my_d = my_minkowski(v1,v2,3)
scipy_d = minkowski(v1,v2,3)
print("minkowski:",my_d)
print("scipyminkowski:",scipy_d)
# A6 : Train–Test split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print("Train:",X_train.shape,"Test:",X_test.shape)

# A7 : kNN k=3

def train_knn(Xtr,ytr,k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Xtr,ytr)
    return model

knn = train_knn(X_train,y_train,3)



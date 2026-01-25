import numpy as np
import pandas as pd
import time

def Time(func,data,runs):
    times=[]
    for _i in range(runs):
          start=time.perf_counter()
          func(data)
          end=time.perf_counter()
          times.append(end-start)
    return sum(times)/runs

df=pd.read_csv("Lab Session Data(IRCTC Stock Price).csv")
def replace_str_num(col):
      return col.astype(str).str.replace(",","").astype(float)

df["Open"]  = replace_str_num(df["Open"])
df["High"]  = replace_str_num(df["High"])
a=df["Open"].values
a=a[0:249]

b=df["High"].values
b=b[0:249]

print("dont produnct of the the 2 data frames ")
print(np.dot(a,b))

print(np.linalg.norm(a)) # lenght of the euclidian norm of a.
print(np.linalg.norm(b)) # lenght of the euclidian norm of the b.

def inn(a):
     return np.linalg.norm(a)
# comparing with the time complexitys 
t_dot=Time(inn,a,10)
print(t_dot)

# 2. 
# function to write the for the mean and the varience .
def mean(col):
      return sum(col)/len(col)
def var(col):
      m=mean(col)
      return sum((col-m)**2)/len(col)

# these functions are used to wrritten for time comparision .
def numpy_var(x):
     return np.var(x)
def numpy_mean(x):
     return np.mean(x)
def my_mean(x):
     return mean(x)
def my_var(x):
     return var(x)

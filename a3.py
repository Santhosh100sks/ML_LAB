import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# this function is used to remove the commmas in the csv file and convermts it into the flot values.
def replace_str_num(col):
      return col.astype(str).str.replace(",","").astype(float)

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

# computational time accuracy b/t methods .
def Time(func,data,runs):
    times=[]
    for _i in range(runs):
          start=time.perf_counter()
          func(data)
          end=time.perf_counter()
          times.append(end-start)
    return sum(times)/runs


df=pd.read_csv("Lab Session Data(IRCTC Stock Price).csv")
print(df)

df["Price"] = replace_str_num(df["Price"])
df["Open"]  = replace_str_num(df["Open"])
df["High"]  = replace_str_num(df["High"])

x=df["Price"].values
x=x[0:249]

y=df["Open"].values
y=y[0:249]

z=df["High"].values
z=z[0:249]

x_mean=np.mean(x)
x_var=np.var(x)
print(x_mean)
print(x_var)
y_mean=np.mean(y)
y_var=np.var(y)
print(y_mean)
print(y_var)
z_mean=np.mean(z)
z_var=np.var(z)
print(z_mean)
print(z_var)


# by using function which was written by me 
xx_mean=mean(x)
xx_var=var(x)
print(x_mean)
print(x_var)
yy_mean=mean(y)
yy_var=var(y)
print(y_mean)
print(y_var)
zz_mean=mean(z)
zz_var=var(z)
print(z_mean)
print(z_var)

# this passes the true bcz we are using this with float 
if np.isclose(xx_mean, x_mean) and np.isclose(yy_mean, y_mean) and np.isclose(zz_mean, z_mean):
    print("Means are matched")
else:
    print("No")
if np.isclose(xx_var, x_var) and np.isclose(yy_var, y_var) and np.isclose(zz_var, z_var):
    print("Variances are matched")
else:
    print("No")


# for computational comparision we need to use time of execution for n runs between the normal python and numpy ,and try to understand between those in detail 
inbuilt_var=Time(numpy_var,x,10)
myb_var=Time(my_var,x,10)
print(f"by inbuilt : {inbuilt_var} , by defind :{myb_var}")
# for mean comparision 
inbuilt_mean=Time(numpy_mean,x,10)
myb_mean=Time(my_mean,x,10)
print(f"by inbuilt : {inbuilt_mean} , by defind :{myb_mean}")

# conclusion i says :
'''The mean and variance calculated using user-defined functions match closely with NumPy’s results, proving correctness.
Although both approaches have O(n) time complexity, NumPy functions execute significantly faster due to optimized C-level implementation and vectorization.'''


# need to rake values for all wednesdays and compare the sample mean and compare the population mean 
wed_p=df.loc[df["Day"]=="Wed","Price"]
print(wed_p)

# mean at the wednesdays is :
m_w=numpy_mean(wed_p)
print(m_w)
# population mean is :
p_mean=numpy_mean(x)
print(p_mean)
# where in this my observations are the mean at the wednesdays are less than compare to the mean of the total prices .

april_p=df.loc[df["Month"]=="Apr","Price"]
m_w1=numpy_mean(april_p)
print(m_w1)

# population mean is :
p_mean1=numpy_mean(x)
print(p_mean1)

# where in the above comparision m_w1 is more than the population mean.
df["Chg%"]=df["Chg%"].astype(str).str.replace("%","").astype(float)
l_flag=df["Chg%"].apply(lambda x: x<0)
prob=l_flag.sum()/len(df)
l=df[df["Chg%"] > 0]
p=len(l)/len(df) # alter way ..
print(f" {prob} , {p}")

# probablity of making profit on wed
wed_flag = df.loc[df["Day"]=="Wed", "Chg%"]  
wed_prob_profit = (wed_flag > 0).sum() / len(wed_flag)
print(f"Prob of making profit on Wed is : {wed_prob_profit}")

profit_wed=(df.loc[df["Day"]=="Wed","Chg%" ]>0).sum()
no_wed=len(df[df["Day"]=="Wed"])
conditional_probability=(profit_wed/no_wed)
print(f"{conditional_probability}")

day_order = {"Mon":1, "Tue":2, "Wed":3, "Thu":4, "Fri":5}
df["DayNum"] = df["Day"].map(day_order)

# Scatter plot
plt.figure(figsize=(10,6))
plt.scatter(df["DayNum"], df["Chg%"], color='blue', alpha=0.7)

# Set x-ticks to show weekday names
plt.xticks([1,2,3,4,5], ["Mon","Tue","Wed","Thu","Fri"])

# Labels and title
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Scatter of Chg% vs Day of the Week")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
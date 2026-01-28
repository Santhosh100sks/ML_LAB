import numpy as np
import pandas as pd

df=pd.read_csv("Lab Session Data(Purchase data).csv")
x=df["Payment (Rs)"].values
poor=x[x<=200]
rich=x[x>200]
print(len(poor))
print(len(rich))

import numpy as np
import pandas as pd
import time

df=pd.read_csv("Lab Session Data(thyroid0387_UCI).csv")
# now based on the iloc values some are in capitals so every thing i am converting onto the small letters . 
df=df.map(lambda x: x.lower() if isinstance(x,str) else x)
binary_df=df.loc[:,df.apply(lambda c: set(c.dropna().unique()).issubset({'f','t'}))]
binary_df=binary_df.replace({'t':1,'f':0})
row1_vector=binary_df.iloc[0].values
row2_vector=binary_df.iloc[1].values
print([[row1_vector]])
l=len(row1_vector)

f11=f10=f01=f00=0
for a,b in zip(row1_vector,row2_vector) :
    if a==1 and b==1:
        f11+=1
    elif a==1 and b==0:
        f10+=1
    elif a==0 and b==0:
        f00+=1
    elif a==0 and b==1:
        f01+=1
jc=f11 / (f01 + f10 + f11)
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)
print(f"jc : {jc} , smc: {SMC} ")

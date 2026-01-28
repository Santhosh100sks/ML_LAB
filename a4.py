import numpy as np
import pandas as pd

df = pd.read_csv("Lab Session Data(thyroid0387_UCI).csv")
df.replace("?", np.nan, inplace=True)
actual_numeric = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
for col in actual_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # where this line is used for the error means ? and non integer values error=coerce
cons_stats = df[actual_numeric].describe().T
cons_stats['variance'] = df[actual_numeric].var()
cons_stats['mean'] = df[actual_numeric].mean()
print(cons_stats['variance'])
cons_stats['range'] = cons_stats['max'] - cons_stats['min']
print(cons_stats['range'])
print(cons_stats['max'])
print(cons_stats['min'])
print(cons_stats['mean'])

def outliers_count(c):
    q1=c.quantile(0.25)
    q2=c.quantile(0.75)
    ior=q2-q1
    lower=q1-1.5*ior
    upper=q2+1.5*ior
    print(ior)
    outliers=c[(c<lower) | (c>upper)]
    return len(outliers)
for i in actual_numeric:
    #each col has no of outliers are :
    n=outliers_count(df[i].dropna())   # this was used to remove the duplicate values .
    print(f"{i} has {n} outliers.")
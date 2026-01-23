import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Lab Session Data(thyroid0387_UCI).csv")

print("DATASET SHAPE")
print(df.shape)
print("\nCOLUMN DATATYPES")
print(df.dtypes)
print("\nMISSING VALUES PER COLUMN")
print(df.isnull().sum())

categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(categorical_cols)
print(numeric_cols)

le = LabelEncoder()

for col in categorical_cols:
    if df[col].nunique() <= 2:
        df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=[col for col in categorical_cols if df[col].dtype == 'object'])


print(df[numeric_cols].agg(['min', 'max']))
print(df[numeric_cols].mean())
print(df[numeric_cols].std())

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
print("\nOUTLIER COUNT PER NUMERIC COLUMN")
print(outliers)

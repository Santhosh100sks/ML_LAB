import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("Lab Session Data(thyroid0387_UCI).csv")
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
print(categorical_cols
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

doc1 = df.iloc[0].select_dtypes(include=[np.number]).values
doc2 = df.iloc[1].select_dtypes(include=[np.number]).values

cosine_similarity = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
print(cosine_similarity)

minmax_scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numeric_cols] = minmax_scaler.fit_transform(df_minmax[numeric_cols])

standard_scaler = StandardScaler()
df_standard = df.copy()
df_standard[numeric_cols] = standard_scaler.fit_transform(df_standard[numeric_cols])

print(df_minmax.head())
print(df_standard.head())

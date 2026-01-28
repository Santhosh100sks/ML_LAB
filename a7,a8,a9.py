import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Lab Session Data(thyroid0387_UCI).csv")
df = df.iloc[:20, :]
df = df.applymap(lambda x: x.lower() if isinstance(x,str) else x)

# A8
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

for col in num_cols:
    if df[col].skew() < 1:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# A7
binary_df = df.loc[:, df.apply(lambda x: set(x.dropna().unique()).issubset({'t','f'}))]
binary_df = binary_df.replace({'t':1,'f':0})
df[binary_df.columns] = binary_df

n = len(df)
JC = np.zeros((n,n))
SMC = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        v1 = df.iloc[i][binary_df.columns]
        v2 = df.iloc[j][binary_df.columns]
        f11=f10=f01=f00=0
        for a,b in zip(v1,v2):
            if a==1 and b==1:
                f11+=1
            elif a==1 and b==0:
                f10+=1
            elif a==0 and b==0:
                f00+=1
            elif a==0 and b==1:
                f01+=1
        JC[i,j] = f11/(f01+f10+f11) if (f01+f10+f11)!=0 else 0
        SMC[i,j] = (f11+f00)/(f00+f01+f10+f11)

numeric_df = df.select_dtypes(include=[np.number])
COS = cosine_similarity(numeric_df)

sns.heatmap(JC)
plt.show()

sns.heatmap(SMC)
plt.show()

sns.heatmap(COS)
plt.show()

# A9
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numeric_df.columns] = scaler.fit_transform(df[numeric_df.columns])
print(df_normalized.head())

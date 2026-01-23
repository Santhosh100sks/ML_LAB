import pandas as pd
import numpy as np

file_path = "Lab Session Data(Purchase data).csv"
df = pd.read_csv(file_path)
print(df)

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values

print(y)

dimensionality = X.shape[1]
num_vectors = X.shape[0]

rank_X = np.linalg.matrix_rank(X)
X_pinv = np.linalg.pinv(X)
cost = X_pinv @ y

print(" vector space:", dimensionality)
print("No.of vectors:", num_vectors)
print("Rank :", rank_X)

print("Candies (Rs):", cost[0])
print("Mangoes (Rs/kg):", cost[1])
print("Milk Packets (Rs):", cost[2])
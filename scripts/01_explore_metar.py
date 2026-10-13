
import  pandas as pd
df=pd.read_csv("data/raw/LTAC.csv", na_values="M")

print(df.shape)
print(df.dtypes)
print(df.head(1))

print(df.iloc[0])
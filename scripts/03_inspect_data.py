import pandas as pd
df=pd.read_csv("data/raw/LTAC_2018_2024.csv",na_values="M")
#print(df.columns.tolist())
#print(df.shape)
#print(df.dtypes)

df["valid"]=pd.to_datetime(df["valid"])
print(len(df) , df["valid"].min() , df["valid"].max())


aylik=df.groupby(df["valid"].dt.to_period("M")).size()
print(aylik.describe())
print(aylik[aylik<1300])
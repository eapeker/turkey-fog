import os
import  pandas as pd
print(os.getcwd())
os.chdir("C:/Users/emina/Desktop/eumetsat/turkey-fog")
df=pd.read_csv("data/raw/LTAC.csv", na_values="M")

#print(df.shape)
print(df.dtypes)
#print(df.head(5))

print(df.iloc[0])
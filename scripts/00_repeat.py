import pandas as pd

df=pd.read_excel("data/raw/repeat/01_repeat.xlsx", na_values="E")

delayed_1=(df["fog"]>6) & (df["rainfall"]>6)
delayed_2=df["fog"]>9
delayed_3=df["rainfall"]>7

delayed=(delayed_1|delayed_2|delayed_3)
print("kalkış yapamayan uçaklar: ", df[delayed]["id"].tolist())

non_delayed=df[~delayed]
print("kalkış yapan uçakların fiyat ortalaması($):",(non_delayed["price"].mean()).round(2))
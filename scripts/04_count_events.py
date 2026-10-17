import pandas as pd

df=pd.read_csv("data/raw/LTAC_2018_2024.csv")
df["valid"]=pd.to_datetime(df["valid"])
#print(df.columns.tolist())
#print(df.dtypes)
#print(df["wxcodes"].unique())

low_vsby=df["vsby"]<0.62

words=df["wxcodes"].fillna("").str.split()
df["is_fog"]=words.apply(lambda w: "FG" in w or "FZFG" in w)
print("sisli rapor sayısı: ",df["is_fog"].sum())

print("-"*80)


fog=df[df["is_fog"]].sort_values("valid")
gap=fog["valid"].diff()
fog["event_id"]=(gap>pd.Timedelta("1h")).cumsum()
print(fog["event_id"].nunique())

print("-"*80)

events=fog.groupby("event_id")["valid"].agg(["min","max","count"])
events["duration"]=events["max"]-events["min"]
print(events.head(10))
print(events["duration"].describe())
print(events["duration"].max(), events["duration"].min())

uzun = events["duration"].idxmax()
print(fog[fog["event_id"] == uzun][["valid", "vsby", "wxcodes"]])
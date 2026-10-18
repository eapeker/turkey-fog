import pandas as pd

def load_metar(path):
    df=pd.read_csv(path,na_values="M",low_memory=False)
    df["valid"]=pd.to_datetime(df["valid"])
    return df

def mark_fog(df):
    words=df["wxcodes"].fillna("").str.split()
    df["is_fog"]=(df["vsby"]<0.62) & words.apply(lambda w: "FG" in w or "FZFG" in w)
    return df


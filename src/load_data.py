import pandas as pd

def load_covid_data(path="data/owid-covid-data.csv", country="India"):
    df = pd.read_csv(path)
    df = df[df["location"] == country][["date", "new_cases"]]
    df['date'] = pd.to_datetime(df['date'])
    df = df.fillna(0)
    return df

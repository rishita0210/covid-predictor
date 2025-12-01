import pandas as pd

def load_covid_data(path="data/weekly-covid-cases.csv"):
    df = pd.read_csv(path)
    return df


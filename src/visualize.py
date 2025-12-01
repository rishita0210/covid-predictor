import matplotlib.pyplot as plt

def plot_cases(df):
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["new_cases"])
    plt.title("COVID-19 Daily New Cases Trend")
    plt.savefig("plots/cases_trend.png")
    plt.close()

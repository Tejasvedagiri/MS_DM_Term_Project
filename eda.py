import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("creditcard.csv")

stats = df.describe()
stats.to_csv("results/stats.csv")

sns.heatmap(df.corr() ,cmap="Blues")
plt.savefig("results/corr.png")

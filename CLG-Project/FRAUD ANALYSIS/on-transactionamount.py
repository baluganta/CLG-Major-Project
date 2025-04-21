import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("up real data1.csv")
df["Fraudulent"] = (df["Transaction Amount (INR)"] > 30000).astype(int)
fraud_counts = df["Fraudulent"].value_counts()

# Pie chart
plt.figure(figsize=(7, 7))
plt.pie(fraud_counts, labels=["Non-Fraud", "Fraud"], autopct="%1.1f%%",
        colors=["green", "red"], startangle=140,
        wedgeprops={'edgecolor': 'black'})
plt.title("Fraud vs Non-Fraud Distribution (Based on Transaction Amount)")
plt.show()

# Bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values,
            palette=["green", "red"])
plt.xticks(ticks=[0, 1], labels=["Non-Fraud", "Fraud"])
plt.xlabel("Fraud Classification")
plt.ylabel("Count")
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

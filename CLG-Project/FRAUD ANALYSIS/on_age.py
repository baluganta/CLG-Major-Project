import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Reload data
df = pd.read_csv("up real data1.csv")

# Check for underage users (fraud if under 18)
df["Fraudulent"] = (df["User Age"] < 18).astype(int)
fraud_counts = df["Fraudulent"].value_counts()

# If both categories exist, show pie chart
if len(fraud_counts) == 2:
    plt.figure(figsize=(7, 7))
    plt.pie(fraud_counts, labels=["Non-Fraud", "Fraud"], autopct="%1.1f%%",
            colors=["yellow", "red"], startangle=140,
            wedgeprops={'edgecolor': 'black'})
    plt.title("Fraud vs Non-Fraud Distribution (Based on Age)")
    plt.show()
else:
    print("Only one category present. Cannot create a pie chart.")
    print(fraud_counts)

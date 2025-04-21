import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reload the dataset
df = pd.read_csv("up real data1.csv")

# Create a new 'Fraudulent' column based on credit score
df["Fraudulent"] = (df["Credit Score"] < 500).astype(int)

# Count of fraud and non-fraud cases
fraud_counts = df["Fraudulent"].value_counts()

# Labels and colors for pie chart
labels = ["Non-Fraud", "Fraud"]
colors = ["green", "blue"]

# Create a pie chart
plt.figure(figsize=(7, 7))
plt.pie(fraud_counts, labels=labels, autopct="%1.1f%%", colors=colors,
        startangle=140, wedgeprops={'edgecolor': 'black'})
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

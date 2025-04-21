import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("up real data1.csv")
num_records = len(df)

# Add synthetic IP address and random fraud labels
df = df.assign(
    Transaction_ID=[f"TXN{i+1}" for i in range(num_records)],
    IP_Address=[f"192.168.{random.randint(0,255)}.{random.randint(0,255)}" for _ in range(num_records)],
    Fraudulent=[random.choice([0, 1]) for _ in range(num_records)]
)

# Count and visualize
fraud_counts = df["Fraudulent"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(fraud_counts, labels=["Non-Fraudulent", "Fraudulent"],
        autopct="%1.1f%%", colors=["lightblue", "red"], startangle=140)
plt.title("Fraudulent vs Non-Fraudulent Transactions")
plt.show()

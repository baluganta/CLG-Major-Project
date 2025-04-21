import pandas as pd

# Load the dataset
df = pd.read_csv('up real data1.csv')

# Display first few rows
df.head()

# Display last few rows
df.tail()

# Summary statistics for numerical columns
df.describe()

# Info on columns, data types, and nulls
df.info()

# List of all column names
df.columns

# Data types of each column
df.dtypes

# Number of unique values per column
df.nunique()

# Shape of the dataset (rows, columns)
df.shape

# View specific rows
df[0:4:1]

# View 'User ID' column for every second record
df[['User ID']][0:8:2]

# Specific value at row 0 and column 'User ID'
df.loc[0,['User ID']]

# Slice first 4 rows and first 4 columns
df.iloc[0:4,0:4]

# Count of missing values per column
df.isnull().sum()

# Count of duplicate rows
df.duplicated().sum()

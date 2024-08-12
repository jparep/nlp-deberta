import pandas as pd
from sklearn.model_selection import train_test_split


# Load Data
real_df = pd.read_csv('/home/jparep/proj/nlp-deberta/data/true.csv')
fake_df = pd.read_csv('/home/jparep/proj/nlp-deberta/data/fake.csv')

# Map label
real_df['label'] = 1
fake_df['label'] = 0

# Concate data
df = pd.concat([real_df, fake_df], ignore_index=True)

# Shuffle and split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

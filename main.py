# Import Libraries
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load Data
real_df = pd.read_csv('/home/jparep/pro/nlp-deberta/data/true.csv')
fake_df = pd.read_csv('/home/jparep/proj/npl-deberta/data/fake.cvs')

# Map labels
real_df['label'] = 1
fake_df['label'] = 0

# Concatinate data
df = pd.concat([real_df, fake_df], axis=0).sample(frac=1).reset_index(drop=True)


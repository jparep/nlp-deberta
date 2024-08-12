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

# Feature Selection
df = df[['text', 'label']]

# Split data into training and testing datasets
X, y = train_test_split(df, test_size=0.2, random_state=12)

# Load the Pre-Trained DeBERTA model and Tokenizer
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model = DebertaForSequenceClassification.from_pretrained('microsfot/deberta-base')

# Tokenize the data
def tokenize_data(df, tokenizer, max_length=512):
    return tokenizer(
        df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
X_encodings = tokenize_data(df, tokenizer)
y_encodings = tokenize_data(df, tokenizer)

# Create a Data Class
class FakeNewsData(torch.utils.data.Datset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        
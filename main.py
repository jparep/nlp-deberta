import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Load Data
real_df = pd.read_csv('/home/jparep/proj/nlp-deberta/data/true.csv')
fake_df = pd.read_csv('/home/jparep/proj/nlp-deberta/data/fake.csv')

# Map label
real_df['label'] = 1
fake_df['label'] = 0

# Concate data
df = pd.concat([real_df, fake_df], axis=0).sample(frac=1).reset_index(drop=True)

# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load DeBERTa Tokenize
tokenizer = DebertaTokenizer.from_pretrained('microsfot/deberta-base')

# Tokenize the data
def tokenize_data(df, tokenizer, max_length=512):
    return tokenizer(
        df['text'].tolist(),
        padding=True,
        trunication=True,
        max_length=max_length,
        return_tensors='pt'
    )

train_encodings = tokenize_data(train_df, tokenizer)
test_encodings = tokenize_data(test_df, tokenizer)

# Create dataset class
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Prepare Datasets
train_dataset = FakeNewsDataset(train_encodings, train_df['label'].tolist())
test_dataset = FakeNewsDataset(test_encodings, test_df['label'].tolist())

# Load the DeBERTa Model
model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base')

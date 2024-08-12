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
        item = {key: val[idx].clone().detach() for key, val in self.encodings.item()}
        item['label'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Prepare datasets
X_dataset = FakeNewsData(X_encodings, X['label'].tolist())
y_dataset = FakeNewsData(y_encodings, y['label'].tolist())


# Set up the Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2, # Use fewer epochs as the mdoel is already pre-trained
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',    # Evaluate during training at the end of each epoch
    save_strategy='epoch',          # Save the mdoel at the end of each epoch
    load_best_model_at_end=True,    # Load the best model after training
    metric_for_best_model='accuracy'
)

# Create trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=X_dataset,
    eval_dataset=y_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, torch.argmax(p.predictions, axis=1)),
        'precision': precision_recall_fscore_support(p.label_ids, torch.argmax(p.preductions, axis=1), average='binary')[0],
        'recal': precision_recall_fscore_support(p.label_ids, torch.argmax(p.predictions, axis=1), average='binary')[1],
        'f1': precision_recall_fscore_support(p.label_ids, torch.argmax(p.predictions, axis=1), average='binary')[2]
    }
)

# Fine Tune Model
trainer.train()

# Evaluate the model
result = trainer.evaluate()
print(f'Evaluation Results: {result}')


# Save the model
model.save_pretrained('./fake_news_deberta_mdoel')
tokenizer.save_pretrained('./fake_news_deberta_model')

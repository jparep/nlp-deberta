import pandas as pd
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Step 1: Load the Data
real_df = pd.read_csv('/home/jparep/proj/nlp-deberta/data/true.csv')
fake_df = pd.read_csv('/home/jparep/proj/nlp-deberta/data/fake.csv')

# Step 2: Add 'label' columns
real_df['label'] = 1
fake_df['label'] = 0

# Step 3: Concatenate and Shuffle the DataFrame
df = pd.concat([real_df, fake_df], axis=0).sample(frac=1).reset_index(drop=True)

# Step 4: Keep Only 'text' and 'label' Columns
df = df[['text', 'label']]

# Step 5: Split the Data into Training and Test Sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 6: Load the Pre-Trained DeBERTa Model and Tokenizer
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base')

# Step 7: Tokenize the Data
def tokenize_data(df, tokenizer, max_length=512):
    return tokenizer(
        df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

train_encodings = tokenize_data(train_df, tokenizer)
test_encodings = tokenize_data(test_df, tokenizer)

# Step 8: Create a Dataset Class
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

# Prepare datasets
train_dataset = FakeNewsDataset(train_encodings, train_df['label'].tolist())
test_dataset = FakeNewsDataset(test_encodings, test_df['label'].tolist())

# Step 9: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,              # Use fewer epochs as the model is already pre-trained
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",     # Evaluate during training at the end of each epoch
    save_strategy="epoch",           # Save the model at the end of each epoch
    load_best_model_at_end=True,     # Load the best model after training
    metric_for_best_model="accuracy"
)

# Step 10: Create Trainer Instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, torch.argmax(p.predictions, axis=1)),
        'precision': precision_recall_fscore_support(p.label_ids, torch.argmax(p.predictions, axis=1), average='binary')[0],
        'recall': precision_recall_fscore_support(p.label_ids, torch.argmax(p.predictions, axis=1), average='binary')[1],
        'f1': precision_recall_fscore_support(p.label_ids, torch.argmax(p.predictions, axis=1), average='binary')[2],
    }
)

# Step 11: Fine-Tune the Model
trainer.train()

# Step 12: Evaluate the Model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Step 13: Save the Model
model.save_pretrained('./fake_news_deberta_model')
tokenizer.save_pretrained('./fake_news_deberta_model')

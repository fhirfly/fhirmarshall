import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
import fhirtorch

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define tokenizer
clinical_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
umls_tokenizer = BertTokenizer.from_pretrained('./umlsbert/config.json')

def tokenize_text(dataframe, column_name, tokenizer):
    text_inputs = dataframe[column_name].astype(str).tolist()
    encoded_text = tokenizer(text_inputs, padding='max_length', truncation=True, max_length=512, return_tensors='pt')   
    return encoded_text['input_ids'], encoded_text['attention_mask'].to(device)

class FHIRDataset(Dataset):
    def __init__(self, texts_clinical, texts_umls, numerical, categorical, targets):
        self.texts_clinical = texts_clinical.to(device)
        self.texts_umls = texts_umls.to(device)
        self.numerical = numerical.to(device)
        self.categorical = categorical.to(device)
        self.targets = targets.to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.texts_clinical[idx], self.texts_umls[idx], self.numerical[idx], self.categorical[idx], self.targets[idx]

class HybridModel(nn.Module):
    def __init__(self, num_features, cat_features):
        super(HybridModel, self).__init__()
        self.bert_clinical = AutoModel.from_pretrained('./clinicalbert/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint/bert_config.json')
        self.bert_umls = AutoModel.from_pretrained('./umlsbert/config.json')
        
        self.fc_text = nn.Linear(self.bert_clinical.config.hidden_size + self.bert_umls.config.hidden_size, 128).to(device)
        self.fc_other = nn.Sequential(
            nn.Linear(num_features + cat_features, 64).to(device),
            nn.ReLU(),
            nn.Linear(64, 32).to(device),
            nn.ReLU(),
            nn.Linear(32, 16).to(device),
            nn.ReLU()
        ).to(device)
        self.fc_out = nn.Sequential(
            nn.Linear(128 + 16, 16).to(device),
            nn.ReLU(),
            nn.Linear(16, 1).to(device)
        ).to(device)

    def forward(self, input_ids_clinical, attention_mask_clinical, input_ids_umls, attention_mask_umls, numerical, categorical):
        text_outputs_clinical = self.bert_clinical(input_ids=input_ids_clinical, attention_mask=attention_mask_clinical).last_hidden_state
        text_outputs_umls = self.bert_umls(input_ids=input_ids_umls, attention_mask=attention_mask_umls).last_hidden_state
        
        text_outputs = torch.cat([text_outputs_clinical[:, 0], text_outputs_umls[:, 0]], dim=1)
        text_outputs = self.fc_text(text_outputs)

        other_outputs = self.fc_other(torch.cat([numerical, categorical], dim=1))

        outputs = self.fc_out(torch.cat([text_outputs, other_outputs], dim=1))

        return outputs

# Load and preprocess the data
with open('fhir_data.json', 'r' ,encoding='utf-8') as f:
    data = fhirtorch.flatten_json(json.load(f))

df = pd.json_normalize(data)
df['birthDate'] = pd.to_datetime(df['birthDate'])
df['birthDate'] = (df['birthDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

boolean_columns, numerical_columns, date_columns, categorical_columns, text_columns, umls_columns = fhirtorch.classify_columns(df)

print('Boolean columns:', boolean_columns)
print('Numerical columns:', numerical_columns)
print('Date columns:', date_columns)
print('Categorical columns:', categorical_columns)
print('Text columns:', text_columns)
print('UMLS columns:', umls_columns)

# Use sklearn's StandardScaler and OneHotEncoder for numerical and categorical data respectively
scaler = StandardScaler()
encoder = OneHotEncoder()

# Replace these lines with your own preprocessing for numerical and categorical data
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
#df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

text_data_umls = tokenize_text(df, df[umls_columns], umls_tokenizer)

# Separate out the text, numerical, and categorical data
for column in df[text_columns]:
    text_data_clinical = text_data_clinical + tokenize_text(df ,column, clinical_tokenizer)

num_data = torch.tensor(df[numerical_columns].values, dtype=torch.float32).to(device)
cat_data = torch.tensor(df[categorical_columns].values, dtype=torch.float32).to(device)
targets = torch.tensor(df['target'].values, dtype=torch.float32).to(device)
targets = torch.tensor(df['target'].values, dtype=torch.float32).to(device)

num_features = len(numerical_columns)
cat_features = df[categorical_columns].nunique().sum()

# Create datasets and dataloaders
train_data, test_data, train_targets, test_targets = train_test_split(np.concatenate([text_data_clinical, text_data_umls, num_data, cat_data], axis=1), targets, test_size=0.2, random_state=42)

dataset = FHIRDataset(train_data[:, :512], train_data[:, 512:1024], train_data[:, 1024:-cat_features], train_data[:, -cat_features:], train_targets)
test_dataset = FHIRDataset(test_data[:, :512], test_data[:, 512:1024], test_data[:, 1024:-cat_features], test_data[:, -cat_features:], test_targets)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

num_features = len(numerical_columns)
cat_features = df[categorical_columns].nunique().sum()

# Initialize the model and optimizer
model = HybridModel(num_features, cat_features)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        texts_clinical, texts_umls, numerical, categorical, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(texts_clinical, texts_umls, numerical, categorical)
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
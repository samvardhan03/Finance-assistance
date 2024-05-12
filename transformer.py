import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Define your dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].values)

# Define data preprocessing steps
def preprocess_data(data):
    # Fill missing values with zeros
    data_filled = data.fillna(0)

    # Convert categorical variables to numerical representations using label encoding
    categorical_cols = data_filled.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data_filled[col] = data_filled[col].astype('category').cat.codes

    # Normalize numerical features
    numeric_cols = data_filled.select_dtypes(include=['int', 'float']).columns
    data_normalized = (data_filled[numeric_cols] - data_filled[numeric_cols].mean()) / data_filled[numeric_cols].std()

    # Combine processed numerical and categorical features
    processed_data = pd.concat([data_normalized, data_filled[categorical_cols]], axis=1)

    return processed_data

# Load the data from CSV files
stock_data = pd.read_csv("/content/drive/My Drive/National_Stock_Exchange_of_India_Ltd.csv")
customer_data = pd.read_csv("/content/drive/My Drive/Customer DataSet.csv")
transaction_data = pd.read_csv("/content/drive/My Drive/bank_transactions.csv")

# Preprocess the loaded data
preprocessed_stock_data = preprocess_data(stock_data)
preprocessed_customer_data = preprocess_data(customer_data)
preprocessed_transaction_data = preprocess_data(transaction_data)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode text data
text_data = ["Example sentence one.", "Another example sentence."]
encoded_data = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')

# Get BERT embeddings for the encoded data
with torch.no_grad():
    outputs = bert_model(input_ids=encoded_data['input_ids'], attention_mask=encoded_data['attention_mask'])
    bert_embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

# Example usage of dataset class
stock_dataset = MyDataset(preprocessed_stock_data)
customer_dataset = MyDataset(preprocessed_customer_data)
transaction_dataset = MyDataset(preprocessed_transaction_data)

# Example usage of DataLoader
stock_loader = DataLoader(stock_dataset, batch_size=32, shuffle=True)
customer_loader = DataLoader(customer_dataset, batch_size=32, shuffle=True)
transaction_loader = DataLoader(transaction_dataset, batch_size=32, shuffle=True)

# Define your model with consistent data types
class FinancialAdviceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FinancialAdviceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x.float())  # Ensure input data type is consistent
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Example usage of the model
model = FinancialAdviceModel(input_size=preprocessed_stock_data.shape[1], hidden_size=64, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop with consistent data types
num_epochs = 10
for epoch in range(num_epochs):
    for batch in stock_loader:
        optimizer.zero_grad()

        # Check if the batch contains valid data
        if torch.any(torch.isnan(batch)):
            continue

        outputs = model(batch)
        loss = criterion(outputs, batch[:, -1].unsqueeze(1).float())  # Ensure target data type is consistent
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), '/content/drive/My Drive/financial_advice_model.pth')


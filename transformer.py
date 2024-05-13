import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

transactions = pd.read_csv('transactions.csv')
customers = pd.read_csv('customer_dataset.csv')
stock_data = pd.read_csv('stock_exchange_data.csv')

# Merge datasets based on CustomerID
merged_data = pd.merge(transactions, customers, on='CustomerID', how='inner')

# Feature engineering
scaler = StandardScaler()
numerical_columns = ['balance', 'balance_frequency', 'purchases', 'ONEOFF_PURCHASES', 'installment_purchases', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'tenure']
merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])

# Merge with stock data based on a common identifier (symbol)
merged_data = pd.merge(merged_data, stock_data, on='symbol', how='inner')

# Define feature columns and target columns
feature_columns = numerical_columns + ['open', 'high', 'low', 'LTP', 'chng', '%chnge', 'volume (lacs)', 'turnover (crs)', '52 w H', '52 wL', '365 d % chng', '30 d % chng']
target_columns = ['target_column1', 'target_column2']

X_train, X_val, y_train, y_val = train_test_split(merged_data[feature_columns], merged_data[target_columns], test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train.values)
y_train_tensor = torch.Tensor(y_train.values)
X_val_tensor = torch.Tensor(X_val.values)
y_val_tensor = torch.Tensor(y_val.values)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.einsum('nqhd,nkhd->nhqk', [Q, K])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        attention = torch.nn.functional.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        x = torch.einsum('nhql,nlhd->nqhd', [attention, V]).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc_out(x)
        return x


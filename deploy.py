import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the feature columns and target columns
feature_columns = ['CustAccountBalance', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                   'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                   'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                   'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
target_columns = ['TransactionAmount (INR)']

# Load the saved model
model = FinancialTransformer(input_size=len(feature_columns), hidden_size=256, output_size=len(target_columns))
model.load_state_dict(torch.load('financial_transformer_model.pth', map_location=torch.device('cpu')))
model.eval()

# Create a label encoder
label_encoder = LabelEncoder()
label_encoder.fit(feature_columns)

# Define the chatbot function
def financial_chatbot(input_text):
    # Preprocess the input text
    input_data = pd.DataFrame([input_text.split(',')], columns=feature_columns)

    # Convert the input data to numerical indices
    input_indices = torch.tensor(label_encoder.transform(input_data.astype(str)), dtype=torch.long)

    # Get the model's output
    with torch.no_grad():
        output = model(input_indices)

    # Convert the output to a human-readable response
    response = f"The predicted transaction amount is: {output.item()} INR"

    return response

# Streamlit app
st.title("Financial Chatbot")
input_text = st.text_input("Enter the feature values (separated by commas):")
if input_text:
    response = financial_chatbot(input_text)
    st.write(response)

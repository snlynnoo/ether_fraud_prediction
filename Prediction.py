# Importing libraries
import streamlit as st 
import pandas as pd
import numpy
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import load

# Setting custom title and BG color
st.set_page_config(
	page_title = "Fraud Prediction",
	page_icon = '🔎'
	)

st.markdown(
	"""
	<style>
	.main {
	background-color: #d2e4f1;
	}
	<style>
	""", unsafe_allow_html=True
)

# Model selection radio button
model_name = st.sidebar.radio(
    'Select Model',
    ('Random Forest', 'XGBoost','Logistic Regression')
	)

header = st.container()

# Into header and image
with header:
	st.header('Ethereum Fraud Prediction System')
	st.image('data/images/cover_photo.jpg')
	st.text('Welcome to Ethereum Fraud Prediction Models\nThe prediction uses (16) features and performing with up to 99% Accuracy\nPlease enter below values in order to predict')

# Input data collection form
with st.form(key='values'):
	input1 = st.number_input('Average minutes between sent transaction')
	input2 = st.number_input('Avgrage minutes between received transaction')
	input3 = st.number_input('Time Difference between first and last in mins')
	input4 = st.number_input('Number of sent transaction')
	input5 = st.number_input('Received total transaction')
	input6 = st.number_input('Number of created contracts')
	input7 = st.number_input('Avgerage value received')
	input8 = st.number_input('Avgerage value sent')
	input9 = st.number_input('Total Ether sent')
	input10 = st.number_input('Total Ether balance')
	input11 = st.number_input('ERC20 total Ether received')
	input12 = st.number_input('ERC20 total Ether sent')
	input13 = st.number_input('ERC20 total Ether sent to contracts')
	input14 = st.number_input('ERC20 unique sent address')
	input15 = st.number_input('ERC20 uniq received token name')

	submited_data = st.form_submit_button(label = 'Predict')

# Variable for input data to feed the model
data_to_predict  = pd.DataFrame(
              ({'Avg min between sent tnx': [input1], 'Avg min between received tnx':[input2],
               'Time Diff between first and last (Mins)': [input3], 
                'Sent tnx': [input4], 'Received Tnx': [input5], 'Number of Created Contracts': [input6],  
                'avg val received': [input7], 'avg val sent': [input8], 'total Ether sent': [input9],
                'total ether balance': [input10],
                'ERC20 total Ether received': [input11], 'ERC20 total ether sent': [input12], 
                'ERC20 total Ether sent contract': [input13],
                'ERC20 uniq sent addr': [input14],'ERC20 uniq rec token name': [input15]}))

# Loading pre-built models
# Loading Random Forest
with open('models/RF.pickle', 'rb') as handle:
    RF = pickle.load(handle, encoding='latin1')
# Loading Logistic Regression
with open('models/LR.pickle', 'rb') as handle:
    LR = pickle.load(handle, encoding='latin1')
# Loading XGBoost 
XGB_C = XGBClassifier()
XGB_C.load_model("models/XGB_C_Final.json")

# Setting model from user selection
def get_model(model_name):
	model = None
	if model_name == 'XGBoost':
		model = XGB_C
	if model_name == 'Random Forest':
		model = RF
	if model_name == 'Logistic Regression':
		model = LR
	return model
selected_model = get_model(model_name)

# Predicting the fraud
result = selected_model.predict(data_to_predict)
result = result[0]

# Interpretation for model result
def predict_result():
	final_result = None
	if result == 0:
		final_result = " ✅ This is NOT a Fraudulent Transaction"
	else:
		final_result = " 🚨 This is a Fraudulent Transaction"
	return final_result
final_result = predict_result()
st.success(final_result)

#===== END =====


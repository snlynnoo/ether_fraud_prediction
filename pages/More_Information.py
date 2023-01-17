import streamlit as st 

st.markdown(
	"""
	<style>
	.main {
	background-color: #d2e4f1;
	}
	<style>
	""", unsafe_allow_html=True
)

section = st.container()
model_comparison = st.container()
references = st.container()

with section:
	st.subheader("About the Project")
	st.image('data/images/info.jpg')
	st.markdown('+ Ethereum is a software platform that uses the concept of blockchain and decentralizes data by distributing copies of smart contracts to thousands of individuals worldwide. Ethereum, as a currency, is utilized to exchange value worldwide in the absence of a third party to monitor or intervene. The popularity of blockchain-based currencies has grown among enthusiasts since 2009. Relying on the anonymity provided by the blockchain, hustlers have adapted offline scams to this new ecosystem.')
	st.markdown('+ This project use supervised machine learning to provide a detection model for Ponzi schemes on Ethereum. It also examines different models such as Random Forest (RF), XGBoost (XGB) and Logistic Regression (LR),for classifying Ethereum fraud detection dataset with and compares their metrics using accuracy and precison scores.')
	st.markdown('+ Random Forest demonstrates the highest accuracies and precision score whereas Logistic Regression shows the lowest performance.')
	st.markdown(' ')
	st.subheader('Models Comparision')
	st.markdown('* **Random Forest (RF):**  Accuracy = 99.0%, Precision = 0.98')
	st.markdown('* **XGBoost (XGB):**  Accuracy = 96.0%, Precision = 0.95')
	st.markdown('* **Logistic Regression:**  Accuracy = 53.0%, Precision = 0.61')
	st.markdown(' ')
	st.subheader('Data Source and References')
	st.markdown('Dataset: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset/code')
	st.markdown('(1) Aziz, Rabia Musheer, et al. “LGBM: A Machine Learning Approach for Ethereum Fraud Detection.” International Journal of Information Technology, 29 Jan. 2022, 10.1007/s41870-022-00864-6.')
	st.markdown('(2) Jung, Eunjin, et al. “Data Mining-Based Ethereum Fraud Detection.” 2019 IEEE International Conference on Blockchain (Blockchain), July 2019, 10.1109/blockchain.2019.00042. Accessed 15 Aug. 2022.')

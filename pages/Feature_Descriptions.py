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

header = st.container()

with header:
	st.header("Descriptions of Features")
	st.markdown(' ')
	st.text('The variables used in predicting the fraud are provided as below.')
	st.markdown(' ')
	st.markdown('🎯 Total Number of features: **16**')
	st.markdown(' ')
	st.markdown('* **Avgerage minutes between sent transaction**\n Average time between received transactions for account in minutes')
	st.markdown(' ')
	st.markdown('* **Avgerage minutes between received transaction**\n Average time between received transactions for account in minutes')
	st.markdown(' ')
	st.markdown('* **Time Difference between first and last in mins**\n Time difference between the first and last transaction')
	st.markdown(' ')
	st.markdown('* **Number of sent transaction**\n Total number of sent normal transactions')
	st.markdown(' ')
	st.markdown('* **Received total transaction**\n Total number of received normal transactions')
	st.markdown(' ')
	st.markdown('* **Number of created contracts**\n Total Number of created contract transactions')
	st.markdown(' ')
	st.markdown('* **Avgerage value receivedv**\n Average value in Ether ever received')
	st.markdown(' ')
	st.markdown('* **Avgerage value sent**\n Average value of Ether ever sent')
	st.markdown(' ')
	st.markdown('* **Total Ether sent**\n Total Ether sent for account address')
	st.markdown(' ')
	st.markdown('* **Total Ether balance**\n Total Ether Balance following enacted transactions')
	st.markdown(' ')
	st.markdown('* **ERC20 total Ether received**\n Total ERC20 token received transactions in Ether')
	st.markdown(' ')
	st.markdown('* **ERC20 total Ether sent**\n Total ERC20 token sent transactions in Ether')
	st.markdown(' ')
	st.markdown('* **ERC20 total Ether sent to contracts**\n Total ERC20 token transfer to other contracts in Ether')
	st.markdown(' ')
	st.markdown('* **RC20 unique sent address**\n Number of ERC20 token transactions sent to Unique account addresses')
	st.markdown(' ')
	st.markdown('* **ERC20 uniq received token name**\n Number of Unique ERC20 tokens received')
	st.markdown(' ')
	st.markdown('**===== End of Descriptions =====**')
	
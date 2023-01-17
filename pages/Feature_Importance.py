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
	st.header("Features Importance")
	st.text('The following figure represents the variables that have the highest influence in\npredicting fraud in decending order.')
	st.markdown(' ')
	st.image('data/images/features_imp.png')


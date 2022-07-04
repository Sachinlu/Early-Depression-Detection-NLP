import streamlit as st
from Program import conv_text as ct

st.title("""Group-10 Early Depression Detection""")
m_selected = st.sidebar.radio("Please select one of the below deep learning model", ['LSTM', 'CNN'])
if m_selected == 'LSTM':
    model = ct.to_select_model('lstm')
    st.write("""LSTM model selected""")
else:
    model = ct.to_select_model('cnn')
    st.write('CNN model selected')

user_input = st.text_input('Enter your tweet')
user_input = [str(user_input)]

if st.button('tweet'):
    tweet = ct.predict_sentiment(user_input, model)
else:
    tweet = 3

st.write("""OUTPUT""")
if tweet == 1:
    st.error('Depressed state')
elif tweet == 0:
    st.success('Non-depressed state')
else:
    st.write("Waiting For Tweet......")

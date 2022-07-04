from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd


x_train = pd.read_csv("DATA/x_train.csv")
x_train = x_train.drop("Unnamed: 0", axis=1)
x_train = x_train.to_numpy()
x_train = x_train.flatten().astype(str)


T = 51

max_vocab = 20000000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(x_train)


#
def to_select_model(val):
    # SELECTING THE MODEL FROM LIST
    if val == 'lstm':
        model_s = load_model('MODEL/lstm_final.h5')
    elif val == 'cnn':
        model_s = load_model('MODEL/cnn_final.h5')
    else:
        pass
    return model_s


#
def predict_sentiment(text, model):
    # preprocessing the given text
    text_seq = tokenizer.texts_to_sequences(text)
    text_pad = pad_sequences(text_seq, maxlen=T)

    # predicting the class
    predicted_sentiment = model.predict(text_pad).round()

    return predicted_sentiment



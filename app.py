import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('nextword_model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Create reverse index
reverse_index = {
    idx: word for word, idx in tokenizer.word_index.items()
}

max_len = 44


def generate_text(seed_text, num_words=10):
    text = seed_text

    for _ in range(num_words):

        # Convert text into sequence
        seq = tokenizer.texts_to_sequences([text])[0]

        # Pad sequence
        padded = pad_sequences(
            [seq],
            maxlen=max_len,
            padding='pre'
        )

        # Predict next word
        preds = model.predict(padded, verbose=0)

        # Get index of predicted word
        pos = np.argmax(preds)

        # Convert index back to word
        next_word = reverse_index.get(pos, "")

        # Add predicted word
        text += " " + next_word

    return text


# Streamlit UI
st.title('Next Word Prediction with Deep Learning')

seed = st.text_input(
    'Enter a Starting Text:',
    'Hello'
)

num_words = st.slider(
    'Number of words to generate',
    1,
    20,
    10
)

if st.button('Generate'):
    result = generate_text(seed, num_words)
    st.write(result)
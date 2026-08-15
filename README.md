# Next Word Prediction System

A deep learning-based text prediction system that predicts the most probable next word based on previously entered text. The project processes a large text corpus, generates sequential training patterns, and learns contextual relationships between words to perform next-word prediction.

## Project Overview

The Next Word Prediction System is an NLP-based project designed to predict the next possible word in a given text sequence.

The project follows a complete pipeline from text preprocessing and tokenization to sequence generation, model training, prediction, and deployment.

The system uses previously entered words as context and predicts the most probable next word based on patterns learned from the training corpus.

## Objectives

- Develop a next-word prediction system for text sequences
- Perform text preprocessing and data cleaning
- Convert textual data into numerical sequences
- Generate n-gram sequences for supervised learning
- Train a deep learning model to learn sequential word patterns
- Predict the most probable next word
- Save the trained model and tokenizer for deployment
- Build an interactive application for text prediction

## Dataset

The project uses the HuffPost News Category Dataset in JSON Lines format.

The original dataset contains 209,527 records and 6 columns:

- `link`
- `headline`
- `category`
- `short_description`
- `authors`
- `date`

For this project, the `short_description` column was selected as the primary text corpus, and the first 20,000 text descriptions were used for model development.

### Dataset Statistics

- Original records: 209,527
- Records used: 20,000
- Text column: `short_description`
- Duplicate records identified: 66
- Duplicate records removed: 66
- Generated sequences: 300,014
- Maximum sequence length: 44
- Vocabulary size: 26,344

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the JSON Lines dataset using Pandas.
2. Selected the `short_description` column.
3. Selected the first 20,000 text descriptions.
4. Checked for missing values.
5. Checked for duplicate text records.
6. Removed duplicate records.
7. Tokenized the text corpus.
8. Added an Out-of-Vocabulary token to handle unseen words.
9. Converted text into numerical sequences.
10. Generated n-gram sequences.
11. Applied pre-padding to standardize sequence length.
12. Separated input sequences from target words.
13. Split the dataset into training and testing sets.

## Text Tokenization

The Keras `Tokenizer` was used to convert words into numerical representations.

An Out-of-Vocabulary token was included to handle words that were not present in the learned vocabulary.

```python
tokenizer = Tokenizer(oov_token='<00V>')
```

The tokenizer creates a word-to-index mapping that allows textual data to be converted into numerical sequences.

For example:

```text
Text:
the government announced

Numerical sequence:
[2, 158, 159]
```

## N-Gram Sequence Generation

N-gram sequences were created from the tokenized text.

For example, a text sequence such as:

```text
I love machine learning
```

can generate sequences such as:

```text
I love
I love machine
I love machine learning
```

Each generated sequence is used to create an input-target pair.

For example:

```text
Input:
I love machine

Target:
learning
```

This allows the model to learn the relationship between previously occurring words and the next word.

## Sequence Padding

Since the generated sequences have different lengths, padding was applied to make all sequences the same length.

The maximum sequence length identified was 44.

Pre-padding was used so that each sequence has a consistent length before being passed to the model.

After padding, the generated dataset contained:

- 300,014 sequences
- 44 tokens per sequence

The input and target values were then separated:

```python
x = input[:, :-1]
y = input[:, -1]
```

This results in:

- Input sequence length: 43
- Target: Next word

## Train-Test Split

The generated sequences were divided into training and testing datasets using an 80:20 split.

```text
Training Data: 80%
Testing Data: 20%
```

The training data is used to learn word relationships, while the testing data is used to evaluate the model's performance on unseen sequences.

## Model Architecture

The project uses a deep learning architecture designed for sequential text prediction.

```text
Input Sequence
      ↓
Embedding Layer
      ↓
LSTM Layer
      ↓
Dense Layer
      ↓
Softmax
      ↓
Predicted Next Word
```

### Embedding Layer

The Embedding layer converts numerical word indices into dense vector representations.

Configuration:

- Vocabulary Size: 26,344
- Embedding Dimension: 128

### LSTM Layer

The LSTM layer learns sequential relationships and contextual dependencies between words.

Configuration:

- LSTM Units: 256

### Dense Layer

The final Dense layer produces a probability distribution over the complete vocabulary.

The word with the highest probability is selected as the predicted next word.

## Prediction Process

The prediction process follows these steps:

```text
User Input
    ↓
Tokenizer
    ↓
Convert Text to Sequence
    ↓
Pad Sequence
    ↓
Model Prediction
    ↓
Probability Distribution
    ↓
Select Highest Probability
    ↓
Predicted Next Word
```

For example:

```text
Input:
The government announced

Predicted Next Word:
new
```

The predicted word can then be added to the existing text and passed back through the model to generate additional words.

## Model Output

The final Dense layer uses Softmax activation to generate probabilities for all words in the vocabulary.

Conceptually, the model produces an output such as:

```text
government    → 0.03
announced     → 0.04
new           → 0.72
policy        → 0.06
```

The word with the highest probability is selected as the next predicted word.

## Streamlit Application

The trained model can be integrated into a Streamlit application to provide an interactive interface for next-word prediction.

The application allows users to:

- Enter starting text
- Select the number of words to generate
- Generate predicted text
- View the generated output interactively

## Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow
- Keras
- Natural Language Processing
- LSTM
- Streamlit
- Jupyter Notebook

## Libraries Used

```python
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

## Key Features

- Text data preprocessing
- Duplicate detection and removal
- Text tokenization
- Out-of-Vocabulary word handling
- Numerical sequence generation
- N-gram sequence generation
- Sequence padding
- Word embeddings
- LSTM-based sequence learning
- Next-word probability prediction
- Model and tokenizer persistence
- Interactive Streamlit deployment

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Next-Word-Prediction-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Next-Word-Prediction-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

## Example Usage

Enter a starting sentence such as:

```text
The president announced
```

The application processes the input text using the saved tokenizer and model and predicts the most probable next word.

The predicted word can be repeatedly added to the input sequence to generate longer text.

## Project Highlights

- Processed 20K+ text descriptions
- Generated 300K+ n-gram sequences
- Built a vocabulary of 26K+ words
- Used a 44-token maximum sequence length
- Implemented 128-dimensional word embeddings
- Trained a 256-unit LSTM network
- Saved the trained model as `nextword_model.h5`
- Saved the tokenizer as `tokenizer.pkl`
- Prepared the model for Streamlit deployment

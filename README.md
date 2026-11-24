# IMDB Movie Review Sentiment Analysis using LSTM

## Project Overview

This project implements a Deep Learning-based Sentiment Analysis system to classify movie reviews as **Positive or Negative** using Long Short-Term Memory (LSTM) networks.

Using the IMDB Movie Reviews dataset, the model processes textual reviews through text preprocessing, word embeddings, and sequential learning to predict the sentiment expressed in each review.

The project explores NLP preprocessing techniques, Bag of Words (BoW), TF-IDF vectorization, and an LSTM-based neural network for binary sentiment classification.

## Objectives

- Perform sentiment classification on movie reviews.
- Apply NLP preprocessing techniques to clean and normalize text.
- Convert textual data into numerical representations.
- Build an LSTM neural network for sequential text classification.
- Evaluate model performance using classification metrics.
- Visualize training performance and prediction results.

## Dataset

**Dataset:** IMDB Movie Reviews Dataset  
**Source:** TensorFlow Keras Datasets

| Attribute | Details |
|---|---|
| Total Reviews | 50,000 |
| Training Samples | 25,000 |
| Testing Samples | 25,000 |
| Classes | Positive, Negative |
| Vocabulary Size | 10,000 |
| Maximum Sequence Length | 250 |

The dataset contains equally distributed positive and negative movie reviews.

## Technologies & Libraries

- Python
- TensorFlow & Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn

## Project Workflow

```text
       IMDB Dataset
             |
             v
       Data Loading
             |
             v
      Review Decoding
             |
             v
       Text Cleaning
             |
             v
     NLP Preprocessing
   (Stopword Removal &
        Stemming)
             |
             v
    Text Representation
       (BoW & TF-IDF)
             |
             v
      Sequence Padding
             |
             v
      LSTM Architecture
             |
             v
       Model Training
             |
             v
      Model Evaluation
             |
             v
    Sentiment Prediction
```

## Data Preprocessing

The following NLP preprocessing techniques were implemented:

1. **Text Cleaning:** Converted text to lowercase and removed HTML tags, special characters, and unnecessary spaces.
2. **Stopword Removal:** Removed common English stopwords using NLTK.
3. **Stemming:** Applied Porter Stemmer to reduce words to their root forms.
4. **Text Vectorization:** Implemented CountVectorizer (BoW) and TfidfVectorizer for feature extraction.
5. **Sequence Padding:** Converted reviews into fixed-length sequences of 250 tokens using Keras padding.

## Model Architecture

An LSTM-based Sequential Neural Network was developed using TensorFlow and Keras.

| Layer | Configuration |
|---|---|
| Embedding | 10,000 input dimensions, 128 output dimensions |
| LSTM | 64 units, 20% dropout |
| Dense | 64 neurons, ReLU activation |
| Dropout | 30% |
| Output | 1 neuron, Sigmoid activation |

**Training Configuration:**

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 128
- Maximum Epochs: 8
- Validation Split: 20%
- Early Stopping: Patience of 2 epochs
- Random Seed: 42

## Model Evaluation

The trained LSTM model was evaluated on 25,000 unseen test reviews.

### Performance Metrics

| Metric | Score |
|---|---:|
| Accuracy | 58.76% |
| Precision | 79.88% |
| Recall | 23.41% |
| F1-Score | 36.21% |
| Test Loss | 0.6706 |

### Evaluation Visualizations

The notebook includes:

- Training vs. Validation Accuracy
- Training vs. Validation Loss
- Confusion Matrix
- Classification Report
- Actual vs. Predicted Sentiment Comparison

## Prediction System

A custom prediction function was implemented to classify new movie reviews.

The function:

1. Accepts a user-provided movie review.
2. Cleans and tokenizes the text.
3. Converts words into numerical sequences.
4. Applies sequence padding.
5. Passes the sequence through the trained LSTM model.
6. Returns the predicted sentiment and positive probability.

## Model Export

The trained model is saved in Keras format:

```python
model.save("imdb_lstm_sentiment_model.keras")
```

## Model Limitations

The current model achieved 58.76% test accuracy, with relatively low recall for positive reviews.

This indicates that the model has difficulty identifying a significant proportion of positive sentiments. Further improvements could include:

- Hyperparameter optimization.
- Experimenting with different LSTM architectures.
- Increasing training epochs with appropriate regularization.
- Comparing LSTM with GRU and Bidirectional LSTM.
- Improving text preprocessing and sequence handling.
- Evaluating alternative embedding approaches.

## Future Enhancements

- Implement Bidirectional LSTM and GRU architectures.
- Experiment with pretrained Word2Vec or GloVe embeddings.
- Compare traditional machine learning models with deep learning approaches.
- Improve prediction reliability through hyperparameter tuning.
- Develop an interactive sentiment analysis application using Flask or Streamlit.

## Author

**Varsha A.**

B.Sc. Artificial Intelligence & Machine Learning

- GitHub: [Varshh-hub](https://github.com/Varshh-hub)
- LinkedIn: [Varsha A.](https://www.linkedin.com/in/varsha-a-4038ab290/)
- Portfolio: [Personal Portfolio](https://varshh-hub.github.io/VARSHA---portfolio/)

---

If you find this project interesting, consider giving the repository a star!

# Sentiment_Analysis-using-LSTM
This is a Python code for sentiment analysis of Twitter data. This code uses the Tweepy library to access the Twitter API, Pandas and NumPy for data processing, NLTK for natural language processing, Scikit-Learn for building the model, and TensorFlow for training and testing the model. The code also uses Matplotlib and Plotly for data visualization.

Dependencies
To run this code, you will need to install the following libraries using pip:

tweepy
scikit-learn
tensorflow
Data
The code uses four datasets:

Twitter_Data.csv
apple-twitter-sentiment-texts.csv
finalSentimentdata2.csv
Tweets.csv

Steps
Import the required libraries.
Load the four datasets into pandas dataframes.
Concatenate the dataframes into a single dataframe.
Clean the text data using regular expressions.
Remove stopwords from the text data.
Perform sentiment analysis on the cleaned data using a TensorFlow model.
Visualize the data using Matplotlib and Plotly.
Functions
wordcount_gen(df, category): Generates a word cloud of the top 50 words in tweets of a given sentiment category.
tweet_to_words(raw_tweet): Cleans the raw tweet text by removing non-letter characters, converting to lowercase, and removing stopwords.
train_model(X_train, y_train, X_test, y_test): Builds and trains a TensorFlow model on the training data and tests the model on the test data.
predict_sentiment(text): Predicts the sentiment of a given text using the trained TensorFlow model.

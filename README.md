# Sentiment_Analysis-using-LSTM
\textbf{Sentiment Analysis of Twitter Data}

This is a Python code for sentiment analysis of Twitter data. This code uses the Tweepy library to access the Twitter API, Pandas and NumPy for data processing, NLTK for natural language processing, Scikit-Learn for building the model, and TensorFlow for training and testing the model. The code also uses Matplotlib and Plotly for data visualization.

\textbf{Dependencies}

To run this code, you will need to install the following libraries using pip:

\begin{itemize}
    \item tweepy
    \item scikit-learn
    \item tensorflow
\end{itemize}

\textbf{Data}

The code uses four datasets:

\begin{itemize}
    \item Twitter\_Data.csv
    \item apple-twitter-sentiment-texts.csv
    \item finalSentimentdata2.csv
    \item Tweets.csv
\end{itemize}

\textbf{Steps}

\begin{enumerate}
    \item Import the required libraries.
    \item Load the four datasets into pandas dataframes.
    \item Concatenate the dataframes into a single dataframe.
    \item Clean the text data using regular expressions.
    \item Remove stopwords from the text data.
    \item Perform sentiment analysis on the cleaned data using a TensorFlow model.
    \item Visualize the data using Matplotlib and Plotly.
\end{enumerate}

\textbf{Functions}

\begin{itemize}
    \item \texttt{wordcount\_gen(df, category)}: Generates a word cloud of the top 50 words in tweets of a given sentiment category.
    \item \texttt{tweet\_to\_words(raw\_tweet)}: Cleans the raw tweet text by removing non-letter characters, converting to lowercase, and removing stopwords.
    \item \texttt{train\_model(X\_train, y\_train, X\_test, y\_test)}: Builds and trains a TensorFlow model on the training data and tests the model on the test data.
    \item \texttt{predict\_sentiment(text)}: Predicts the sentiment of a given text using the trained TensorFlow model.
\end{itemize}


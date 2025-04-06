import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pandas as pd
import string
import re

#download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

def drop_na(dataframe):
    """
    Removes rows with null values from the DataFrame.

    :param dataframe: Pandas DataFrame to be cleaned.
    :return: DataFrame without null values.
    """
    return dataframe.dropna()

def tokenize(dataframe, column_name):
    """
    Tokenizes text in a specified column and creates a new column for tokenized text.

    :param dataframe: Pandas DataFrame containing text data.
    :param column_name: Column name containing text to be tokenized.
    :return: DataFrame with an added 'tokenized_text' column.
    """
    dataframe['tokenized_text'] = dataframe[column_name].apply(word_tokenize)
    return dataframe

def stop_words(dataframe, column_name):
    """
    Removes stopwords from tokenized text.

    :param dataframe: Pandas DataFrame containing tokenized text.
    :param column_name: Column name containing tokenized words.
    :return: DataFrame with an added 'no_stop_words' column.
    """
    stop_words = set(stopwords.words('english'))
    dataframe['no_stop_words'] = dataframe[column_name].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
    return dataframe


def lemmatize(dataframe, column_name):
    """
    Lemmatizes words in a specified column using their part of speech.

    :param dataframe: Pandas DataFrame containing tokenized text.
    :param column_name: Column name containing tokenized words.
    :return: DataFrame with an added 'lemmatized_text' column.
    """

    def get_wordnet_pos(word):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)  #default is just noun

    lemmatizer = WordNetLemmatizer()
    dataframe['lemmatized_text'] = dataframe[column_name].apply(lambda words: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words])
    return dataframe

def clean_text(text):
    """
    Remove all punctuation (including smart quotes)
    :param text: pandas row with string
    :return: string with punctuation and emojis removed and multiple white spaces turned into single whitespace
    """
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()

path = "cleaned_reduced.csv"

df = pd.read_csv(path)
df.columns = ['raw_text']
print(df.shape)
print(df.head(10))
print()

#drop na
drop_na(df)

#lowercase everything
df['raw_text'] = df['raw_text'].str.lower()
print(df.shape)
print(df.head(10))
print()

tokenize(df, 'raw_text')
print(df.shape)
print(df.head(10))
print()

#remove stop words
stop_words(df, 'tokenized_text')
print(df.shape)
print(df.head(10))
print()

#lemmatize
lemmatize(df, 'no_stop_words')
print(df.shape)
print(df.head(10))
print()

df['list_to_string'] = df['lemmatized_text'].apply(lambda x: ' '.join(x))
print(df.shape)
print(df.head(10))
print()

df['list_to_string'] = df['list_to_string'].apply(clean_text)
print(df.shape)
print(df.head(10))
print()
df.to_csv('cleaned.csv', index=False)
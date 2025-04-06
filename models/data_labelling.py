import pandas as pd
from transformers import pipeline, DistilBertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder

def labeller(text, neutral_threshold=0.2):
    """
    Labels the given text as POSITIVE, NEGATIVE, or NEUTRAL using a pre-trained sentiment analysis model.

    :param text: Input text to be classified.
    :param neutral_threshold: Confidence threshold below which the text is classified as NEUTRAL.
    :return: Sentiment label (POSITIVE, NEGATIVE, or NEUTRAL).
    """
    try:
        #load sentiment analyser
        classifier = pipeline('sentiment-analysis')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        #tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=False)

        #check if the tokenized input exceeds the maximum length
        if inputs['input_ids'].shape[1] > 512:  #exceeds the max tokens
            return "TOO MANY TOKENS"

        result = classifier(text)[0]
        score = result['score']
        label = result['label'].upper()

        #low confidence scores are classed as neutral
        if score < neutral_threshold:
            return "NEUTRAL"
        return label
    except Exception as e:
        #return the name of the exception as the label
        return f"ERROR: {type(e).__name__}"


def vader_labeller(text, neutral_threshold=0.2):
    """
    Labels the given text as POSITIVE, NEGATIVE, or NEUTRAL using VADER sentiment analysis.

    :param text: Input text to be classified.
    :param neutral_threshold: Confidence threshold below which the text is classified as NEUTRAL.
    :return: Sentiment label (POSITIVE, NEGATIVE, or NEUTRAL).
    """
    try:
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Get the sentiment scores
        sentiment_score = analyzer.polarity_scores(text)

        # Check sentiment score for classification
        if sentiment_score['compound'] >= neutral_threshold:
            return "POSITIVE"
        elif sentiment_score['compound'] <= -neutral_threshold:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    except Exception as e:
        # Return the exception name as the label
        return f"ERROR: {type(e).__name__}"


def tfidf(text):
    """
        Converts a list of text inputs into a TF-IDF matrix and returns a pandas DataFrame.

        :param text: List of textual data.
        :return: DataFrame with TF-IDF values, where columns represent feature names.
    """
    try:
        #tf-idf converts the text info to vector format
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text)

        '''basically tfidf creates multiple features for each word so we will store it in a new pandas dataframe so its easy
        to apply chi-squared later and each word will have its own column'''

        #for the columns of the df
        feature_names = vectorizer.get_feature_names_out()

        return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    except Exception as e:
        return f"Error in tfidf function: {str(e)}"

def chi_squared(X, y, top_n=20):
    """
        Computes chi-squared scores for feature selection and returns the top correlated features.

        :param X: Feature matrix (TF-IDF representation of text).
        :param y: Target labels (POSITIVE, NEGATIVE, etc.).
        :param top_n: Number of top features to return based on chi-squared score.
        :return: DataFrame of top_n features sorted by chi-squared score.
    """
    try:
        #chi2_scores tells us the strength of the correlation to the label higher=more imp, p_value tells us if that correlation is statistically significant lower=strongly related higher=noise
        chi2_scores, p_values = chi2(X, y)

        #storing the data in a pandas df
        chi2_results = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi2_scores, 'P-value': p_values})
        chi2_results = chi2_results[chi2_results['P-value'] < 0.05] #filtering via p-values before sorting by chi2

        return chi2_results.sort_values(by='Chi2 Score', ascending=False)
    except Exception as e:
        return f"Error in chi_squared function: {str(e)}"


def prep_for_ml(df, top_n=20):
    """
    Prepares DataFrame for ML by applying TF-IDF and Chi-squared feature selection using custom functions.

    Parameters:
    - df: DataFrame with 'cleaned_data' (text) and 'sentiment' (labels) columns
    - top_n: Number of top features to select with Chi-squared (default 20)

    Returns:
    - X: Processed text features (TF-IDF + Chi2 selected)
    - y: Encoded sentiment labels
    """
    # Check required columns
    if 'cleaned_data' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("DataFrame must have 'cleaned_data' and 'sentiment' columns")

    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])

    # Print mapping for reference
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Label encoding mapping (sentiment -> number):")
    for sentiment, number in label_mapping.items():
        print(f"{sentiment} -> {number}")

    # Apply your tfidf function to 'cleaned_data'
    X_tfidf = tfidf(df['cleaned_data'])
    if isinstance(X_tfidf, str):  # Check for error
        raise ValueError(X_tfidf)

    # Apply your chi_squared function
    chi2_results = chi_squared(X_tfidf, y, top_n=top_n)
    if isinstance(chi2_results, str):  # Check for error
        raise ValueError(chi2_results)

    # Select top features from TF-IDF matrix
    top_features = chi2_results['Feature'].tolist()
    X = X_tfidf[top_features]

    # Print some info
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    print(f"After Chi2 selection, X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Top {top_n} features selected: {top_features}")

    return X, y

if __name__ == '__main__':
    print("Data labelling and feature selection for sentiment analysis")


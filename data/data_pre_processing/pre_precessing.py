#from data_cleaning import drop_na, tokenize, stop_words, lemmatize
from data_labelling import labeller, tfidf, chi_squared, vader_labeller
import pandas as pd

#path = "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\combined_df.csv"
path = "cleaned.csv"
df = pd.read_csv(path)
print(df.shape)

# Label data with VADER
df['sentiment'] = df['raw_text'].apply(lambda x: vader_labeller(x))

# Rename list_to_string to cleaned_data
df = df.rename(columns={'list_to_string': 'cleaned_data'})

# Select only the 3 desired columns
df_final = df[['raw_text', 'cleaned_data', 'sentiment']]

# Print sentiment distribution
print("\nSentiment counts:")
print(df_final['sentiment'].value_counts())

# Save to CSV
df_final.to_csv("processed_labelled_data.csv", index=False)
print("Final shape:", df_final.shape)
print("Saved DataFrame with columns:", df_final.columns.tolist())
print("\nFirst few rows:")
print(df_final.head())

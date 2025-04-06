import pandas as pd
from data_labelling import prep_for_ml

path = "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_pre_processing\\processed_labelled_data.csv"
df = pd.read_csv(path)
X, y = prep_for_ml(df)
print(X)
print(y)
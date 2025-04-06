import pandas as pd
import os

# List of CSV file paths to combine
csv_paths = [
    "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\web_scraping\\scraping_with_api\\reddit_threads.csv",
    "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\web_scraping\\scraping\\combined_scraped_texts.csv",
    "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\pdf_scraping\\scraped_pdf.csv",
    "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\speech_to_text\\all_transcripts.csv"
]
df = pd.read_csv("C:\\Users\\avyan\PycharmProjects\SentimentAnalysis\data\data_collection\master_df.csv")
# Extract text from each CSV and combine
all_texts = df.values.tolist()

def reddit():
    df = pd.read_csv("C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\web_scraping\\scraping_with_api\\reddit_threads2.csv")
    df['Reddit Threads'] = df['Reddit Threads'].replace("\n","")
    list = df['Reddit Threads'].tolist()
    for text in list:
        text.strip().replace('\n', "")
        all_texts.append(text)

def combined_web():
    df = pd.read_csv("C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\web_scraping\\scraping\\combined_scraped_texts.csv")
    list2 = df['Text'].tolist()
    list2.pop(0)
    for text in list2:
        text.strip().replace('\n', "")
        all_texts.append(text)

def pdf():
    df = pd.read_csv("C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\pdf_scraping\\scraped_pdf.csv")
    df = df.iloc[:, 1:2]
    list3 = df['0'].tolist()
    for text in list3:
        text.strip().replace('\n',"")
        all_texts.append(text)

def whisper():
    df = pd.read_csv("C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\speech_to_text\\all_transcripts.csv")
    list4 = df['text'].tolist()
    for text in list4:
        text.strip().replace('\n', "")
        all_texts.append(text)

print(len(all_texts))
reddit()
print(len(all_texts))
master_df = pd.DataFrame(all_texts)

master_df.to_csv("master_df.csv", index=False)
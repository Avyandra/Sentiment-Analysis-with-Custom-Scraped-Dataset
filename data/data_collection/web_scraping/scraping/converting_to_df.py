import pandas as pd
import os

def combine_csv_files(csv_folder_path):
    """
    This function combines multiple CSV files containing scraped text into one DataFrame.
    Each line of the CSV files will be a row in the final DataFrame.

    :param csv_folder_path: Path to the folder containing the CSV files.
    :return: A pandas DataFrame containing all the text from all files.
    """
    #create an empty list to store data from each CSV
    all_texts = []

    #iterate through all files in the specified folder
    for filename in os.listdir(csv_folder_path):
        if filename.endswith('.csv'):  #process only csv files
            file_path = os.path.join(csv_folder_path, filename)

            # Read the current CSV file
            df = pd.read_csv(file_path, header=None)

            #assuming the text is in the first column (adjust if needed)
            for text in df[0]:
                if isinstance(text, str):  #avoid non-string entries
                    all_texts.append(text.strip())  #remove any extra leading/trailing spaces

    #convert the list of texts into a DataFrame
    combined_df = pd.DataFrame(all_texts, columns=["Text"])

    #drop any empty rows if present
    combined_df = combined_df[combined_df["Text"].str.strip() != ""]

    return combined_df

csv_folder_path = "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\web_scraping\\scraping\\scraped_csv"
df = combine_csv_files(csv_folder_path)

#show the first 10 rows of the combined DataFrame
print(df.head(10))

#save the combined dataframe as a CSV
df.to_csv("combined_scraped_texts.csv", index=False)

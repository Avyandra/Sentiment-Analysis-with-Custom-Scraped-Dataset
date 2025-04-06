from pdfminer.high_level import extract_text
import pandas as pd

def get_pdf_text(pdf_path, pages=None):
    """
    Extracts text from a PDF file.

    :param pdf_path: Path to the PDF file.
    :param pages: Optional list of page numbers to extract (1-based index).
    :return: List of strings, each containing the text of a specified page.
    """
    if pages is None:
        return [extract_text(pdf_path)]

    extracted_text = []

    for page_number in pages:
        text = extract_text(pdf_path, page_numbers=[page_number - 1]) #pdfminer syntax
        extracted_text.append(text.strip())

    return pd.DataFrame(extracted_text)

pdf_path = 'C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_collection\\pdf_scraping\\pdf_files\\TeslaFSD_AnanalysisofYouTubecommentarydrives.pdf'
df = get_pdf_text(pdf_path, pages=(3, 5, 6, 7, 8, 9 ,13, 14, 15))
print(df.shape)
print(df.head(10))
df.to_csv("scraped_pdf.csv")
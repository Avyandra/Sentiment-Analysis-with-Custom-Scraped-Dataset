<<<<<<< HEAD
# Sentiment-Analysis-with-Custom-Scraped-Dataset-
I built and trained my own sentiment analysis models on a custom dataset that was created via web-scraping, reddit scraping, pdf scraping and converting audio to text. This particular project conducted sentiment analysis on Tesla more specifically their FSD system but the code is very reusable and modular. Feel free to try it out.
=======
# Sentiment Analysis with a Custom-Built Dataset from Web, Audio & Documents

## 1. Project Overview

This project demonstrates an **end-to-end sentiment analysis pipeline**, focusing not just on model training, but also **data engineering, preprocessing, feature extraction**, and **evaluation**. We built a **custom dataset** by collecting data from four diverse sources:

- Web scraping (Selenium + BeautifulSoup)
- Reddit API (PRAW)
- PDF text extraction (PDFMiner6)
- Audio transcription (Whisper)

**Sentiment analysis** is the process of detecting emotional tone behind pieces of text. Here, we classify data into **Positive**, **Neutral**, or **Negative** sentiments.

---

## 2. Data Collection

We designed custom scripts for each data source:

### • Web Scraping
- **What**: Automated extraction of website content
- **Tools**: `Selenium` for rendering dynamic content, `BeautifulSoup` for HTML parsing  
- **Use Case**: News articles and blog posts

*Placeholder: [Data sources pipeline diagram]*  
*Placeholder: [Selenium + BeautifulSoup architecture diagram]*

### • Reddit API
- **What**: Extracting post content from Reddit
- **Tools**: `PRAW` API  
- **Use Case**: Topical opinions and informal discussions

### • PDF Scraping
- **What**: Extracting raw text from PDFs
- **Tools**: `PDFMiner6`  
- **Use Case**: Reports, whitepapers, and formal documents

### • Speech-to-Text
- **What**: Transcribing audio to text
- **Tools**: OpenAI's `Whisper`  
- **Use Case**: Conversations, podcasts, and voice notes

---

## 3. Data Preprocessing

### a. Cleaning

Steps applied to all raw data:

- Merged into a single **Pandas DataFrame**
- Dropped nulls
- **Tokenized** text (split into words) using `nltk`
- Removed **stopwords** (common but meaningless words)
- **Lemmatized** words (reduced to root form)
- Removed punctuation via regex and joined tokens
- Trimmed whitespace

*Placeholder: [Raw vs Cleaned CSV samples image]*

### b. Labeling

- Used **VADER** sentiment analyzer (ideal for raw text)
- Automatically labeled data into:
  - `Positive`
  - `Neutral`
  - `Negative`
- Final DataFrame contains: `raw_text`, `clean_text`, `sentiment`

### c. TF-IDF & Feature Selection

- Built a function to:
  - Convert `clean_text` to **TF-IDF vectors**
  - Encode labels with `LabelEncoder`
  - Apply **Chi-squared test** for relevant feature selection

*Placeholder: [TF-IDF matrix snapshot]*

---

## 4. Model Training

Trained and validated the following classifiers using `scikit-learn`:

- **Logistic Regression**  
  - Multinomial, 70/30 train-test split  
  - Chosen for simplicity and interpretability

- **Multinomial Naive Bayes**  
  - Good baseline for text classification

- **Random Forest**  
  - 100 estimators, handles non-linear patterns well

- **Support Vector Machine (SVM)**  
  - Linear kernel, `decision_function_shape='ovr'`  
  - Good with high-dimensional sparse data (like TF-IDF)

All models were tuned for **multiclass classification**.

---

## 5. Evaluation

Used a separate script to evaluate each model:

- **Confusion Matrix**: Visualizes prediction errors
- **Classification Report**: Displays accuracy, precision, recall, and F1-score per class

**Tools**: `matplotlib`, `seaborn`

*Placeholder: [Confusion matrix plots]*  
*Placeholder: [Classification report snapshots]*

---

## 6. Model Comparison

Final script loads pickled models and compares performance:

- Metrics:
  - Accuracy
  - Positive F1
  - Neutral F1
  - Negative F1
- Output: Bar chart comparison of all models

*Placeholder: [Final model comparison chart]*

---

## 7. Visuals

*Image placeholders included throughout the repo:*

- Data sources pipeline (smart art)
- Web scraping architecture
- Raw vs cleaned CSV comparison
- TF-IDF matrix visual
- Confusion matrices (for all models)
- Final model comparison chart

---

## Get Started

1. Clone the repo
2. Install dependencies with `pip install -r requirements.txt`
3. Run data collection scripts in `data_collection/`
4. Preprocess with `preprocessing/cleaning.py`
5. Train models in `models/train.py`
6. Evaluate with `evaluation/eval.py`

---

## License

MIT License – Feel free to use or adapt with credit.

---

## Contributions

Open to PRs. Please open an issue first if you want to contribute a major change.
>>>>>>> origin/master

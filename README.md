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

<img width="329" alt="data_gathering_pipeline" src="https://github.com/user-attachments/assets/1728e847-2d95-4104-81a7-8873544c5b12" />

### • Web Scraping
- **What**: Automated extraction of website content
- **Tools**: `Selenium` for rendering dynamic content, `BeautifulSoup` for HTML parsing  
- **Use Case**: News articles and blog posts

<img width="614" alt="web_scraping_1" src="https://github.com/user-attachments/assets/16270ed7-5dd5-4415-a0aa-234d4ece6eef" />
<img width="313" alt="web_scraping_pipeline" src="https://github.com/user-attachments/assets/d1965923-08d7-4c69-879f-a1541bc2a688" />


### • Reddit API
- **What**: Extracting post content from Reddit
- **Tools**: `PRAW` API  
- **Use Case**: Topical opinions and informal discussions

<img width="482" alt="reddit_scraping_1" src="https://github.com/user-attachments/assets/15010128-3b5e-4ab2-acb6-07ca267b5bcd" />

### • PDF Scraping
- **What**: Extracting raw text from PDFs
- **Tools**: `PDFMiner6`  
- **Use Case**: Reports, whitepapers, and formal documents

<img width="461" alt="pdf_scraping_1" src="https://github.com/user-attachments/assets/3f6e321d-7fae-4c08-b162-c416fdb4c0a7" />

### • Speech-to-Text
- **What**: Transcribing audio to text
- **Tools**: OpenAI's `Whisper`  
- **Use Case**: Conversations, podcasts, and voice notes
<img width="474" alt="whisper_1" src="https://github.com/user-attachments/assets/3d3d684d-085a-47b5-9e05-291aff838493" />
<img width="602" alt="whisper_2" src="https://github.com/user-attachments/assets/cc0c419c-d3ae-4584-bd9b-010ad4f05521" />

---

## 3. Data Preprocessing

<img width="281" alt="data_pre_processing" src="https://github.com/user-attachments/assets/9aa0f5de-a125-41bd-925d-3f497c29b68c" />

### a. Cleaning

Steps applied to all raw data:

- Merged into a single **Pandas DataFrame**
- Dropped nulls
- **Tokenized** text (split into words) using `nltk`
- Removed **stopwords** (common but meaningless words)
- **Lemmatized** words (reduced to root form)
- Removed punctuation via regex and joined tokens
- Trimmed whitespace

<img width="934" alt="data_cleaning_1" src="https://github.com/user-attachments/assets/e503701b-fe60-4b74-ac4d-12692786a4ad" />

### b. Labeling

- Used **VADER** sentiment analyzer (ideal for raw text)
- Automatically labeled data into:
  - `Positive`
  - `Neutral`
  - `Negative`
- Final DataFrame contains: `raw_text`, `clean_text`, `sentiment`

<img width="676" alt="final_df_1" src="https://github.com/user-attachments/assets/4befc08a-a85d-4016-b734-b2b7a00f0ab0" />

### c. TF-IDF & Feature Selection

- Built a function to:
  - Convert `clean_text` to **TF-IDF vectors**
  - Encode labels with `LabelEncoder`
  - Apply **Chi-squared test** for relevant feature selection

<img width="898" alt="tfidf_and_ch2_demo" src="https://github.com/user-attachments/assets/b5d2082a-2540-4738-9807-018502319155" />

---

## 4. Model Training

Trained and validated the following classifiers using `scikit-learn`:

- **Logistic Regression**  
  - Multinomial, 70/30 train-test split  
  - Chosen for simplicity and interpretability
    
<img width="319" alt="Logistic_Regression_Confusion_Matrix" src="https://github.com/user-attachments/assets/6b56fe32-506c-42a6-92b7-0dcde9420d00" />
<img width="500" alt="Logistic_Regression_Heatmap" src="https://github.com/user-attachments/assets/19c5fecc-33b3-4494-bdd2-28b538a2113b" />

- **Multinomial Naive Bayes**  
  - Good baseline for text classification
    
<img width="319" alt="Naive_Bayes_Confusion_Matrix" src="https://github.com/user-attachments/assets/74565bfc-7e50-4535-baa4-68a83011a7bf" />
<img width="499" alt="Naive_Bayes_Heatmap" src="https://github.com/user-attachments/assets/f68f778a-3bdf-46bb-9aea-aa8f991f4c96" />

- **Random Forest**  
  - 100 estimators, handles non-linear patterns well
    
<img width="319" alt="Random_Forest_Confusion_Matrix" src="https://github.com/user-attachments/assets/8d4204ca-7d83-44bf-a73e-7abc1da88f76" />
<img width="499" alt="Random_Forest_Heatmap" src="https://github.com/user-attachments/assets/2b8b3614-40f2-4e56-8f00-62ff8863dfe9" />

- **Support Vector Machine (SVM)**  
  - Linear kernel, `decision_function_shape='ovr'`  
  - Good with high-dimensional sparse data (like TF-IDF)
    
<img width="320" alt="SVM_Confusion_Matrix" src="https://github.com/user-attachments/assets/9903c654-edac-47a6-9c53-26e4f76e6b0a" />
<img width="499" alt="SVM_Heatmap" src="https://github.com/user-attachments/assets/af2706b3-4a44-4b60-ad17-e11c54cf12ee" />

All models were tuned for **multiclass classification**.

---

## 5. Evaluation

Used a separate script to evaluate each model:

- **Confusion Matrix**: Visualizes prediction errors
- **Classification Report**: Displays accuracy, precision, recall, and F1-score per class

**Tools**: `matplotlib`, `seaborn`

<img width="479" alt="eval1" src="https://github.com/user-attachments/assets/e5cb74c2-c0d2-4b1f-81af-6ca24413b116" />
<img width="476" alt="eval2" src="https://github.com/user-attachments/assets/775c85e8-96df-41ce-abb6-c70266cda128" />

---

## 6. Model Comparison

Final script loads pickled models and compares performance:

- Metrics:
  - Accuracy
  - Positive F1
  - Neutral F1
  - Negative F1
- Output: Bar chart comparison of all models

<img width="599" alt="All_Model_Comparison" src="https://github.com/user-attachments/assets/1856b007-7514-4a30-b3ff-cb17f6b1a7bc" />

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

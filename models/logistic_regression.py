from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from evaluation_metrics import plot_confusion_matrix, plot_classification_metrics, save_results
from data_labelling import prep_for_ml
import pandas as pd

#initializing the labels
path = "C:\\Users\\avyan\\PycharmProjects\\SentimentAnalysis\\data\\data_pre_processing\\processed_labelled_data.csv"
df = pd.read_csv(path)
X, y = prep_for_ml(df)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42) #stratify=all classes are evenly distributed, random_state=the data is not shuffled each time

#initializing the linearsvc (multiclass classification svm using one-vs-the-rest)
clf = LogisticRegression(multi_class='multinomial', random_state=0) #can set to ovr too but multinomial is generally better with tfidf, max_iter can be set higher if getting more convergence warnings
clf.fit(X, y) #x=features aka text, y=labels

#testing the models prediction
y_pred = clf.predict(X_test)

#evaluation the test results
plot_confusion_matrix(clf, X_test, y_test)
plot_classification_metrics(y_test, y_pred)
save_results('logistic_regression_results.pkl', y_test, y_pred)




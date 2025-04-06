from sklearn.ensemble import RandomForestClassifier
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

#initializing the random forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)  #n_estimators=number of branches
clf.fit(X_train, y_train)

#testing the models prediction
y_pred = clf.predict(X_test)

#evaluation the test results
plot_confusion_matrix(clf, X_test, y_test)
plot_classification_metrics(y_test, y_pred)
save_results('random_forest_results.pkl', y_test, y_pred)

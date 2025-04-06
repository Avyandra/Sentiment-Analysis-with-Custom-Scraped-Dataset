import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import pandas as pd
import seaborn as sns
import pickle

def plot_confusion_matrix(classifier, X_test, y_test, display_labels=None,
                          normalize='true', cmap='Blues', figsize=(8, 6)):
    """
    Plot a confusion matrix heatmap for a classifier's performance.

    Parameters:
    - classifier: Trained classifier model
    - X_test: Test features
    - y_test: True test labels
    - display_labels: List of class names (e.g., ['Positive', 'Negative', 'Neutral'])
    - normalize: 'true', 'pred', 'all', or None (default: 'true')
    - cmap: Colormap for heatmap (default: 'Blues')
    - figsize: Tuple for figure size (width, height)
    """
    # Set up figure
    plt.figure(figsize=figsize)

    #plot confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=display_labels,
        cmap=plt.cm.get_cmap(cmap),
        normalize=normalize,
    )

    #title of the plot
    disp.ax_.set_title(f'Confusion Matrix (Normalized: {normalize})', pad=15)

    plt.tight_layout()
    plt.show()

def plot_classification_metrics(y_test, y_pred, classes=('Positive', 'Negative', 'Neutral'), figsize=(10, 6)):
    """
    Generate and visualize classification metrics (precision, recall, f1-score).

    Parameters:
    - y_test: True labels
    - y_pred: Predicted labels
    - classes: List of class names
    - figsize: Tuple for figure size (width, height)
    """
    #accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    #classification report as dictionary
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    #convert to pandas dataframe and extract relevant metrics
    df_report = pd.DataFrame(report).T #.T for transpose=flipping row and column
    df_metrics = df_report.loc[classes, ['precision', 'recall', 'f1-score']]

    #size of plot
    plt.figure(figsize=figsize)

    #heatmap
    sns.heatmap(df_metrics, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title(f'Classification Metrics Heatmap (Accuracy: {accuracy:.3f})')
    plt.ylabel('Classes')
    plt.xlabel('Metrics')

    plt.tight_layout()
    plt.show()

def save_results(filename, y_test, y_pred):
    """
    Save model predictions and true labels to a file using pickle.

    Parameters:
    - filename (str): The name of the file to save results to (e.g., 'model_results.pkl').
                      If the file already exists, it will be overwritten.
    - y_test (list or array): The true labels from the test set.
    - y_pred (list or array): The predicted labels from the model.

    Returns:
    - None: This function doesn’t return anything; it saves data to a file and prints a confirmation.

    Notes:
    - Uses binary write mode ('wb'), so the file is saved in a compact, machine-readable format.
    - Overwrites any existing file with the same name.
    """
    #we only need y_test and y_pred to calculate metrics
    results = {'y_true': y_test, 'y_pred': y_pred}
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"{filename} results saved.")

if __name__ == '__main__':
    print("Evaluation metrics functions")
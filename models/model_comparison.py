import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

#define labels
classes = ['Negative', 'Neutral', 'Positive']

#load metrics from each model
models = {
    'Logistic Regression': 'logistic_regression_results.pkl',
    'Random Forest': 'random_forest_results.pkl',
    'Naive Bayes': 'naive_bayes_results.pkl',
    'Supported Vector Machine': 'svc_results.pkl'
}

def main():
    df_metrics = get_metrics()
    plot_graph(df_metrics)
    # summary table
    print("\nModel Comparison Summary:")
    print(df_metrics.round(3).to_string(index=False))

def get_metrics():
    #to store the metrics
    metrics_data = []

    for model_name, file_path in models.items():
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        y_true = results['y_true']
        y_pred = results['y_pred']

        #calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

        #store metrics
        metrics_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Positive F1': report['Positive']['f1-score'],
            'Negative F1': report['Negative']['f1-score'],
            'Neutral F1': report['Neutral']['f1-score']
        })

    # convert to pandas dataframe for easier plotting
    return pd.DataFrame(metrics_data)

def plot_graph(df_metrics):
    metrics = ['Accuracy', 'Positive F1', 'Negative F1', 'Neutral F1']
    n_models = len(models)
    n_metrics = len(metrics)
    bar_width = 0.15
    index = range(n_models)

    plt.figure(figsize=(12, 6))

    #plot bars for each metric
    for i, metric in enumerate(metrics):
        plt.bar([x + bar_width * i for x in index], df_metrics[metric],
                bar_width, label=metric)

    #labels
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison: Key Metrics')
    plt.xticks([i + bar_width * (n_metrics - 1) / 2 for i in index], df_metrics['Model'], rotation=15)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
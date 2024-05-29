import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE

def process(path):
    df = pd.read_csv(path)
    print(df.head())

    # Drop rows with missing values if necessary
    df = df.dropna()

    # Assuming the target variable is 'status' where 1 indicates Parkinson's disease presence
    df['status'] = df['status'].map({0: 0, 1: 1})

    X = df.drop(['name', 'status'], axis=1)  # Features
    y = df['status']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Initialize RandomForestClassifier and RFE
    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    rfe = RFE(estimator=rf_classifier, n_features_to_select=10, step=1)

    # Fit RFE on training data
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # Train RandomForestClassifier on selected features
    rf_classifier.fit(X_train_rfe, y_train)
    y_pred = rf_classifier.predict(X_test_rfe)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("MSE:", mse)
    print("MAE:", mae)
    print("R-squared:", r2)
    print("Accuracy:", accuracy)

    # Save predicted values to CSV
    result_file = "static/results/resultRF.csv"
    with open(result_file, "w") as result_csv:
        result_csv.write("ID,Predicted Value\n")
        for idx, val in enumerate(y_pred, start=1):
            result_csv.write(f"{idx},{val}\n")

    # Save evaluation metrics to CSV
    metrics_file = 'static/results/RFMetrics.csv'
    with open(metrics_file, 'w') as metrics_csv:
        metrics_csv.write("Parameter,Value\n")
        metrics_csv.write(f"MSE,{mse}\n")
        metrics_csv.write(f"MAE,{mae}\n")
        metrics_csv.write(f"R-squared,{r2}\n")
        metrics_csv.write(f"Accuracy,{accuracy}\n")

    # Plot and save bar chart for metrics
    acc = [mse, mae, r2, accuracy]
    alc = ["MSE", "MAE", "R-squared", "Accuracy"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig = plt.figure()
    plt.bar(alc, acc, color=colors)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title('Random Forest Metrics Value')
    fig.savefig('static/results/RFMetricsValueBarChart.png')
    plt.pause(5)
    plt.show(block=False)
    plt.close()


# Call the function with the Parkinson's dataset path
#process("parkinsons.csv")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.svm import SVC

def process(path):
    # Read the dataset
    df = pd.read_csv(path)
    print(df.head())

    # Drop rows with missing values
    df = df.dropna()

    # Assuming 'status' is the target column for classification (0 for healthy, 1 for Parkinson's)
    X = df.drop(['status', 'name'], axis=1).values
    y = df['status'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize the SVM classifier
    model = SVC()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("---------------------------------------------------------")
    print(f"MSE VALUE FOR SVM IS {mse}")
    print(f"MAE VALUE FOR SVM IS {mae}")
    print(f"R-SQUARED VALUE FOR SVM IS {r2}")
    print(f"ACCURACY VALUE SVM IS {accuracy}")
    print("---------------------------------------------------------")

    # Save evaluation metrics to a CSV file
    metrics_file = 'static/results/SVMMetrics.csv'
    with open(metrics_file, 'w') as result2:
        result2.write("Parameter,Value\n")
        result2.write(f"MSE,{mse}\n")
        result2.write(f"MAE,{mae}\n")
        result2.write(f"R-SQUARED,{r2}\n")
        result2.write(f"ACCURACY,{accuracy}\n")

    # Read the metrics CSV file for visualization
    df_metrics =  pd.read_csv('static/results/SVMMetrics.csv')
    acc = df_metrics["Value"]
    alc = df_metrics["Parameter"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]

    # Plot the evaluation metrics
    fig = plt.figure()
    plt.bar(alc, acc, color=colors)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title('SVM Metrics Value')
    fig.savefig('static/results/SVMMetricsValueBarChart.png')
    plt.pause(5)
    plt.show(block=False)
    plt.close()

# Call the function with the dataset path
#process("parkinsons.csv")

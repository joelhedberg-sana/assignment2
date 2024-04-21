# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import json
import sqlalchemy
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Constants and Configuration
MYSQL_HOST = "mysql-1.cda.hhs.se"
MYSQL_USERNAME = "be903"
MYSQL_PASSWORD = "robots"
MYSQL_SCHEMA = "Survivability"
CONNECTION_STRING = (
    f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_SCHEMA}"
)


def connect_to_database():
    """Establish a connection to the MySQL database with error handling."""
    try:
        return sqlalchemy.create_engine(CONNECTION_STRING)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def fetch_data(cnx, table_name):
    """Fetch data from the database, accessing each table only once."""
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, cnx)
    return df


# Question 1: Data-Wrangling & Basic Data Presentation
def process_wbc_data(exam_df):
    """Process White Blood Cell (WBC) data for Question 1.1."""
    wbc_df = exam_df[exam_df["measurement"] == "White Blood Cell Count"]
    return wbc_df


def plot_wbc_distribution(wbc_df):
    """Plot the distribution of White Blood Cell counts for Question 1.1."""
    plt.figure(figsize=(8, 5))
    plt.hist(wbc_df["value"].astype(float), bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("White Blood Cell Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of White Blood Cell Counts")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


# Question 2: Accurate Predictions
def prepare_data_for_model(patient_df, exam_df, study_df):
    """Prepare and merge data for the model."""
    # Perform one-hot encoding on 'language' column
    language_dummies = pd.get_dummies(patient_df["language"], prefix="language")
    patient_df = pd.concat([patient_df, language_dummies], axis=1).drop(
        "language", axis=1
    )

    # Convert 'gender' to numeric
    patient_df["gender"] = patient_df["gender"].map({"male": 0, "female": 1})

    # Convert 'admission_date' to days since a reference date
    reference_date = datetime(2017, 1, 1)
    patient_df["admission_date"] = pd.to_datetime(patient_df["admission_date"])
    patient_df["admission_date"] = (
        patient_df["admission_date"] - reference_date
    ).dt.days

    # Handle 'do_not_resuscitate' column
    patient_df["do_not_resuscitate"] = (
        patient_df["do_not_resuscitate"].fillna(-1).astype(int)
    )

    # Extract features from PatientExamination
    measurements = [
        "Arterial Blood PH",
        "Bilirubin Level",
        "Blood Urea Nitrogen",
        "Glucose",
        "Heart Rate",
        "Mean Arterial Blood Pressure",
        "Respiration Rate",
        "Serum Albumin",
        "Serum Creatinine Level",
        "White Blood Cell Count",
    ]
    extracted_features = pd.DataFrame(index=exam_df["patient_id"].unique())
    for measurement in measurements:
        temp_df = exam_df[exam_df["measurement"] == measurement].pivot(
            index="patient_id", columns="measurement", values="value"
        )
        extracted_features = extracted_features.join(temp_df, how="outer")
    extracted_features.columns = [
        col.lower().replace(" ", "_") for col in extracted_features.columns
    ]
    extracted_features.reset_index(inplace=True)
    extracted_features.rename(columns={"index": "patient_id"}, inplace=True)

    # Merge all data
    df = pd.merge(
        patient_df, extracted_features, left_on="id", right_on="patient_id", how="left"
    )
    df = pd.merge(df, study_df, left_on="id", right_on="patient_id", how="left")

    return df


def train_and_evaluate_model(df):
    """Train the model and evaluate its performance."""
    features = [
        col
        for col in df.columns
        if col
        not in [
            "id",
            "patient_id",
            "died_in_hospital",
            "death_recorded_after_hospital_discharge",
        ]
    ]
    X = df[features]
    y = df["died_in_hospital"]

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model, X.columns.tolist(), X_test, y_test


def update_json_with_predictions(json_data, predictions, key_name):
    if isinstance(json_data, list):
        for i, patient in enumerate(json_data):
            if i < len(predictions):
                patient[key_name] = int(predictions[i])
    else:
        print("json_data is not in the expected format.")
    return json_data


# Question 3: Applied Learnings and Building a Case
def maximize_profit(model, X_test):
    """Maximize KindCorp's profit by offloading claims to EvilCorp."""
    probabilities = model.predict_proba(X_test)[:, 1]
    expected_profits = 500000 * probabilities - 150000
    sorted_indices = np.argsort(expected_profits)[::-1]
    cumulative_profits = np.cumsum(expected_profits[sorted_indices])
    optimal_index = np.argmax(cumulative_profits)
    optimal_threshold = probabilities[sorted_indices][optimal_index]
    classifications = (probabilities >= optimal_threshold).astype(int)
    return classifications, optimal_threshold


def plot_cumulative_profit(model, X_test):
    """Plot the cumulative profit curve."""
    probabilities = model.predict_proba(X_test)[:, 1]
    expected_profits = 500000 * probabilities - 150000
    sorted_indices = np.argsort(expected_profits)[::-1]
    cumulative_profits = np.cumsum(expected_profits[sorted_indices])
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(cumulative_profits)), cumulative_profits, color="skyblue")
    plt.xlabel("Number of Claims Offloaded")
    plt.ylabel("Cumulative Profit (€)")
    plt.title("Cumulative Profit Curve")
    plt.grid(alpha=0.75)
    plt.show()


def calculate_costs(model, X_test, y_test):
    """Calculate the costs for different strategies."""
    cost_of_doing_nothing = -500000 * y_test.sum()
    cost_of_selling_all = -150000 * len(y_test)
    classifications, _ = maximize_profit(model, X_test)
    cost_of_profit_model = (
        -500000 * (y_test * (1 - classifications)).sum()
        - 150000 * classifications.sum()
    )
    savings = cost_of_profit_model - min(cost_of_doing_nothing, cost_of_selling_all)
    return cost_of_doing_nothing, cost_of_selling_all, cost_of_profit_model, savings


def plot_sensitivity_analysis(model, X_test, y_test):
    """Plot the sensitivity analysis of profitability."""
    error_rates = np.linspace(0, 1, 100)
    profits = []
    for error_rate in error_rates:
        # Simulate predictions with the given error rate
        simulated_predictions = model.predict(X_test)
        simulated_predictions[y_test == 0] = np.random.choice(
            [0, 1], size=np.sum(y_test == 0), p=[1 - error_rate, error_rate]
        )
        simulated_predictions[y_test == 1] = np.random.choice(
            [0, 1], size=np.sum(y_test == 1), p=[error_rate, 1 - error_rate]
        )

        # Calculate profit based on simulated predictions
        profit = (
            -500000 * (y_test * (1 - simulated_predictions)).sum()
            - 150000 * simulated_predictions.sum()
        )
        profits.append(profit)

    plt.figure(figsize=(8, 5))
    plt.plot(error_rates, profits, color="skyblue")
    plt.xlabel("Error Rate")
    plt.ylabel("Profit (€)")
    plt.title("Sensitivity Analysis of Profitability")
    plt.grid(alpha=0.75)
    plt.show()


def main():
    cnx = connect_to_database()
    if cnx:
        # Fetch data from each table once
        exam_df = fetch_data(cnx, "PatientExamination")
        patient_df = fetch_data(cnx, "Patient")
        study_df = fetch_data(cnx, "Study")

        # Process and plot WBC data for Question 1.1
        wbc_df = process_wbc_data(exam_df)
        plot_wbc_distribution(wbc_df)

        # Prepare data and train model for Question 2
        df = prepare_data_for_model(patient_df, exam_df, study_df)

    # Load patient IDs from the JSON file
    with open("assignment2.json", "r") as f:
        json_data = json.load(f)
    patient_ids = [patient["patient_id"] for patient in json_data]

    # Filter the dataframe based on patient IDs in the JSON file
    df_filtered = df[df["id"].isin(patient_ids)]

    model, features, X_test, y_test = train_and_evaluate_model(df_filtered)

    # Evaluate model performance for Question 2.2
    predictions = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    # Plot confusion matrix with labels in the cells for better readability
    plt.matshow(confusion_matrix(y_test, predictions), cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, confusion_matrix(y_test, predictions)[i, j], ha='center', va='center', color='black')
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    


    print("\nClassification Report:\n", classification_report(y_test, predictions))

    # Calculate Type I and Type II errors
    type1_error = np.sum((y_test == 0) & (predictions == 1)) / np.sum(
        y_test == 0
    )  # False Positive Rate
    type2_error = np.sum((y_test == 1) & (predictions == 0)) / np.sum(
        y_test == 1
    )  # False Negative Rate

    print("Type I Error (False Positive Rate): ", type1_error)
    print("Type II Error (False Negative Rate): ", type2_error)

    # Identify and print the five most important features for Question 2.4
    encoded_features = features
    if len(encoded_features) == len(model.feature_importances_):
        feature_importances = pd.DataFrame(
            {"feature": encoded_features, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("Top 5 Important Features:\n", feature_importances.head())
    else:
        print("Length of encoded_features array: ", len(encoded_features))
        print(
            "Length of model.feature_importances_ array: ",
            len(model.feature_importances_),
        )
        print(
            "The lengths of the encoded_features and model.feature_importances_ arrays do not match."
        )

    # Update JSON file with predictions for Question 2.3
    updated_json = update_json_with_predictions(
        json_data, predictions, "prediction_accuracy_model"
    )
    with open("assignment2_updated.json", "w") as f:
        json.dump(updated_json, f, indent=4)

    # Question 3.1: Maximize profit and present model performance
    classifications, optimal_threshold = maximize_profit(model, X_test)
    plot_cumulative_profit(model, X_test)
    print(f"Optimal threshold for maximizing profit: {optimal_threshold:.2f}")

    # Question 3.2: Update JSON file with profitability classifications
    updated_json = update_json_with_predictions(
        json_data, classifications, "prediction_profit_model"
    )
    with open("assignment2_updated.json", "w") as f:
        json.dump(updated_json, f, indent=4)

    # Question 3.3: Calculate and present costs for different strategies
    cost_of_doing_nothing, cost_of_selling_all, cost_of_profit_model, savings = (
        calculate_costs(model, X_test, y_test)
    )
    print("Costs for different strategies:")
    print(f"Cost of Doing Nothing: €{-cost_of_doing_nothing/1e6:.2f} million")
    print(f"Cost of Selling All Claims: €{-cost_of_selling_all/1e6:.2f} million")
    print(
        f"Cost When Using Profit Maximization Model: €{-cost_of_profit_model/1e6:.2f} million"
    )
    print(f"Estimated Savings to KindCorp: €{savings/1e6:.2f} million")

    # Question 3.4: Sensitivity analysis of profitability
    plot_sensitivity_analysis(model, X_test, y_test)


if __name__ == "__main__":
    main()

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import json
import sqlalchemy
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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

def identify_admitted_patients(patient_df):
    """Identify patients currently admitted and in need of classification."""
    admitted_df = patient_df[patient_df["days_before_discharge"] > 0]
    return admitted_df["id"].tolist()

def save_patient_ids_to_json(patient_ids, filename="classification_patients.json"):
    """Save the patient IDs to a JSON file."""
    data = [{"patient_id": pid} for pid in patient_ids]
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

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

    # Remove rows with missing values
    print("Number of patients before removing missing values:", df.shape[0])
    #df = df.dropna()
    print("Number of patients after removing missing values:", df.shape[0])

    return df

def train_and_evaluate_model(df):
    """Train the model and evaluate its performance with enhanced checks for feature relevance and model complexity."""
    # Define the features to be included in the model
    features = [
        col
        for col in df.columns
        if col
        not in [
            "id",
            "patient_id",
            "died_in_hospital",
            "death_recorded_after_hospital_discharge",
            "language",
            "admission_date",
            "patient_id_y",
            "patient_id_x",
            "language_Estonian",
            "language_Finnish",
            "language_Other",
            "language_Russian",
            "language_Swedish",
        ]
    ]

    # Prepare the feature matrix X and the target vector y
    X = df[features]
    y = df["died_in_hospital"]

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the RandomForestClassifier with parameters to reduce overfitting
    model = RandomForestClassifier(
        max_depth=5,  # Limiting depth of the tree
        max_features="sqrt",
        min_samples_leaf=10,  # Minimum samples at a leaf
        min_samples_split=20,  # Minimum samples to split a node
        n_estimators=50,
        random_state=42,
        class_weight="balanced",  # Adjust class weights to handle imbalanced data
    )
    model.fit(X_train, y_train)

    # Evaluate model using cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())

    # Feature importance analysis
    feature_importances = pd.DataFrame(
        model.feature_importances_, index=X_train.columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    print("Feature importances:\n", feature_importances)

    # Model evaluation on the test set
    predictions = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    return model, X.columns.tolist(), X_test, y_test


def update_json_with_predictions(json_data, predictions, wbc_counts, key_name):
    if isinstance(json_data, list):
        for i, patient in enumerate(json_data):
            if i < len(predictions):
                patient[key_name] = int(predictions[i])
                if pd.notna(wbc_counts[i]):
                    patient["white_blood_cell_count"] = round(float(wbc_counts[i]))
                else:
                    patient["white_blood_cell_count"] = None
            else:
                patient[key_name] = None  # Set missing predictions to None
                patient["white_blood_cell_count"] = None
    else:
        print("json_data is not in the expected format.")
    return json_data


#Question 3: Applied Learnings and Building a Case
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
    if cnx is not None:
       # Fetch data from each table once and print their column names
        patient_df = fetch_data(cnx, "Patient")
        admitted_patient_ids = identify_admitted_patients(patient_df)
        save_patient_ids_to_json(admitted_patient_ids)
        print(f"Admitted patient IDs saved to classification_patients.json")

        exam_df = fetch_data(cnx, "PatientExamination")
        study_df = fetch_data(cnx, "Study")

        # Filter data to include only admitted patients
        admitted_patient_df = patient_df[patient_df["id"].isin(admitted_patient_ids)]
        admitted_exam_df = exam_df[exam_df["patient_id"].isin(admitted_patient_ids)]
        admitted_study_df = study_df[study_df["patient_id"].isin(admitted_patient_ids)]

        # Process and plot WBC data for Question 1.1
        wbc_df = process_wbc_data(admitted_exam_df)
        plot_wbc_distribution(wbc_df)

        # Prepare data and train model for Question 2
        df = prepare_data_for_model(admitted_patient_df, admitted_exam_df, admitted_study_df)

        # Load patient IDs from the JSON file
        with open("classification_patients.json", "r") as f:
            json_data = json.load(f)
        patient_ids = [patient["patient_id"] for patient in json_data]

        # Filter the dataframe based on patient IDs in the JSON file
        df_filtered = df[df["id"].isin(patient_ids)]
        print("Number of patients in the filtered dataframe:", df_filtered.shape[0])


        # Extract WBC counts for the filtered patients
        wbc_counts = df_filtered["white_blood_cell_count"].tolist()

        model_results = train_and_evaluate_model(df_filtered)

        if model_results is None:
            print("Insufficient samples for training and evaluation. Skipping further analysis.")
            return

        model, features, X_test, y_test = model_results

        # Evaluate model performance for Question 2.2
        predictions = model.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
        print("\nClassification Report:\n", classification_report(y_test, predictions))

        # Update JSON file with predictions and WBC counts for Question 2.3
        updated_json = update_json_with_predictions(
            json_data, predictions, wbc_counts, "prediction_accuracy_model"
        )
        with open("assignment2_updated.json", "w") as f:
            json.dump(updated_json, f, indent=4)

        # Question 3.1: Maximize profit and present model performance
        classifications, optimal_threshold = maximize_profit(model, X_test)
        plot_cumulative_profit(model, X_test)
        print(f"Optimal threshold for maximizing profit: {optimal_threshold:.2f}")

        # Question 3.2: Update JSON file with profitability classifications and WBC counts
        updated_json = update_json_with_predictions(
            json_data, classifications, wbc_counts, "prediction_profit_model"
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

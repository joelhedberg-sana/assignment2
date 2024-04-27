# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import json
import sqlalchemy
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


def identify_admitted_patients(cnx):
    """Identify patients currently admitted and in need of classification."""
    query = """
        SELECT p.id AS patient_id
        FROM Patient p
        LEFT JOIN Study s ON p.id = s.patient_id
        LEFT JOIN PatientExamination pe ON p.id = pe.patient_id
        WHERE p.days_before_discharge > 0
          AND pe.patient_id IS NOT NULL
          AND p.admission_date IS NOT NULL
    """
    admitted_df = pd.read_sql(query, cnx)
    return admitted_df


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


# Plot the distribution of White Blood Cell counts for Question 1.1
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

    # One-hot encode 'disease_category' and 'disease_class'
    disease_category_dummies = pd.get_dummies(
        patient_df["disease_category"], prefix="disease_category"
    )
    disease_class_dummies = pd.get_dummies(
        patient_df["disease_class"], prefix="disease_class"
    )
    patient_df = pd.concat(
        [patient_df, disease_category_dummies, disease_class_dummies], axis=1
    ).drop(["disease_category", "disease_class"], axis=1)

    # Convert 'do_not_resuscitate' to numeric
    patient_df["do_not_resuscitate"] = (
        patient_df["do_not_resuscitate"].fillna(0).astype(int)
    )

    # One-hot encode 'income_bracket'
    income_bracket_dummies = pd.get_dummies(
        study_df["income_bracket"], prefix="income_bracket"
    )
    study_df = pd.concat([study_df, income_bracket_dummies], axis=1).drop(
        "income_bracket", axis=1
    )

    # Create a copy of exam_df to avoid SettingWithCopyWarning
    exam_df_copy = exam_df.copy()

    # Convert 'Has Cancer', 'Has Dementia', and 'Has Diabetes' to numeric
    for measurement in ["Has Cancer", "Has Dementia", "Has Diabetes"]:
        if measurement in exam_df_copy["measurement"].unique():
            if measurement == "Has Cancer":
                exam_df_copy["has_cancer_no"] = (
                    exam_df_copy[exam_df_copy["measurement"] == measurement]["value"]
                    == "no"
                ).astype(int)
                exam_df_copy["has_cancer_yes"] = (
                    exam_df_copy[exam_df_copy["measurement"] == measurement]["value"]
                    == "yes"
                ).astype(int)
                exam_df_copy["has_cancer_metastatic"] = (
                    exam_df_copy[exam_df_copy["measurement"] == measurement]["value"]
                    == "metastatic"
                ).astype(int)
            else:
                exam_df_copy.loc[
                    exam_df_copy["measurement"] == measurement, measurement
                ] = (
                    exam_df_copy.loc[
                        exam_df_copy["measurement"] == measurement, "value"
                    ]
                    .fillna(0)
                    .astype(int)
                )
        else:
            print(f"Warning: {measurement} not found in PatientExamination table.")
            exam_df_copy[measurement] = np.nan

    # Extract features from PatientExamination
    measurements = [
        "Arterial Blood PH",
        "Bilirubin Level",
        "Blood Urea Nitrogen",
        "Glucose",
        "Has Dementia",
        "Has Diabetes",
        "Heart Rate",
        "Mean Arterial Blood Pressure",
        "Number of Comorbidities",
        "P/F Ratio",
        "Respiration Rate",
        "Serum Albumin",
        "Serum Creatinine Level",
        "Serum Sodium Concentration",
        "SPS Score",
        "Temperature",
        "Urine Output",
        "White Blood Cell Count",
    ]

    extracted_features = pd.DataFrame(index=exam_df_copy["patient_id"].unique())
    for measurement in measurements:
        temp_df = exam_df_copy[exam_df_copy["measurement"] == measurement].pivot(
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

    # Merge 'Has Cancer' columns
    df = pd.merge(
        df,
        exam_df_copy.groupby("patient_id")[
            ["has_cancer_no", "has_cancer_yes", "has_cancer_metastatic"]
        ].max(),
        left_on="id",
        right_index=True,
        how="left",
    )
    # Exclude columns related to post-hospital visit data
    post_visit_columns = [
        "death_recorded_after_hospital_discharge",
        "days_of_follow_up",
    ]
    df = df.drop(columns=post_visit_columns, errors="ignore")

    return df


# Question 2: Model Training and Evaluation
def train_and_evaluate_model(df):
    """Train the model and evaluate its performance with enhanced checks for feature relevance and model complexity."""
    # Define the features to be included in the model
    features = [
        "age",
        "gender",
        "doctors_2_months_survival_prediction",
        "doctors_6_months_survival_prediction",
        "do_not_resuscitate",
        "years_of_education",
        "existing_models_2_months_survival_prediction",
        "existing_models_6_months_survival_prediction",
        "days_of_follow_up",
        "days_in_hospital_before_study",
        "arterial_blood_ph",
        "bilirubin_level",
        "blood_urea_nitrogen",
        "glucose",
        "has_cancer_no",
        "has_cancer_yes",
        "has_cancer_metastatic",
        "has_dementia",
        "has_diabetes",
        "heart_rate",
        "mean_arterial_blood_pressure",
        "number_of_comorbidities",
        "p/f_ratio",
        "respiration_rate",
        "serum_albumin",
        "serum_creatinine_level",
        "serum_sodium_concentration",
        "sps_score",
        "temperature",
        "urine_output",
        "white_blood_cell_count",
    ] + [
        col
        for col in df.columns
        if col.startswith(
            ("language_", "disease_category_", "disease_class_", "income_bracket_")
        )
    ]
    # Exclude post-hospital visit columns from the features
    post_visit_columns = [
        "death_recorded_after_hospital_discharge",
        "days_of_follow_up",
    ]
    features = [col for col in features if col not in post_visit_columns]

    # Prepare the feature matrix X and the target vector y
    X = df[features]
    y = df["died_in_hospital"]

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
    print("Feature importances:\n", feature_importances.head())

    return model, features, X_test, y_test

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print("True Negative (TN) | False Positive (FP)")
    print("False Negative (FN)| True Positive (TP)")
    print(cm)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    # Include number of samples in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="red")
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def calculate_error_rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    type_1_error_rate = fp / (fp + tn)
    type_2_error_rate = fn / (fn + tp)
    return type_1_error_rate, type_2_error_rate


# Update JSON file with predictions and WBC counts for Question 2.3 and 3.2
def update_json_with_predictions(
    json_data,
    predictions,
    profit_predictions,
    wbc_counts,
    accuracy_key_name,
    profit_key_name,
):
    if isinstance(json_data, list):
        for i, patient in enumerate(json_data):
            if i < len(predictions):
                patient[accuracy_key_name] = int(predictions[i])
            else:
                patient[accuracy_key_name] = None  # Set missing predictions to None

            if i < len(profit_predictions):
                patient[profit_key_name] = int(profit_predictions[i])
            else:
                patient[profit_key_name] = None  # Set missing predictions to None

            if i < len(wbc_counts) and pd.notna(wbc_counts[i]):
                patient["white_blood_cell_count"] = round(float(wbc_counts[i]))
            else:
                patient["white_blood_cell_count"] = None
    else:
        print("json_data is not in the expected format.")
    return json_data


# Question 3: Maximizing Profit
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


# Plot the cumulative profit curve for Question 3.1
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


# Calculate costs for different strategies for Question 3.3
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


# Plot the sensitivity analysis of profitability for Question 3.4
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


# Main function to execute the entire pipeline
def main():
    cnx = connect_to_database()
    if cnx is not None:
        patient_df = fetch_data(cnx, "Patient")
        admitted_patient_df = identify_admitted_patients(
            cnx
        )  # Pass cnx instead of patient_df
        admitted_patient_ids = admitted_patient_df["patient_id"].tolist()
        save_patient_ids_to_json(admitted_patient_ids)
        print(f"Admitted patient IDs saved to classification_patients.json")
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
        df = prepare_data_for_model(
            admitted_patient_df, admitted_exam_df, admitted_study_df
        )

        # Load patient IDs from the JSON file
        with open("classification_patients.json", "r") as f:
            json_data = json.load(f)
        patient_ids = [patient["patient_id"] for patient in json_data]

        # Filter the dataframe based on patient IDs in the JSON file
        df_filtered = df[df["id"].isin(patient_ids)]
        print("Number of patients in the filtered dataframe:", df_filtered.shape[0])

        # Extract WBC counts for the filtered patients
        wbc_counts = df_filtered.loc[
            df_filtered["id"].isin(patient_ids), "white_blood_cell_count"
        ].tolist()

        # Train and evaluate the model
        model, features, X_test, y_test = train_and_evaluate_model(df_filtered)
        print("Selected features:", features)

        # Evaluate model performance for Question 2.2
        predictions = model.predict(X_test)
        print_confusion_matrix(y_test, predictions)
        type_1_error_rate, type_2_error_rate = calculate_error_rates(y_test, predictions)
        print("Type 1 Error Rate:", type_1_error_rate)
        print("Type 2 Error Rate:", type_2_error_rate)
        print("\nClassification Report:\n", classification_report(y_test, predictions))
        plot_confusion_matrix(y_test, predictions)

        # Maximize profit and get profitability-based classifications for Question 3.2
        profit_classifications, optimal_threshold = maximize_profit(model, X_test)
        print("Optimal threshold:", optimal_threshold)

        # Update JSON file with predictions and WBC counts for Question 2.3 and 3.2
        updated_json = update_json_with_predictions(
            json_data,
            predictions,
            profit_classifications,
            wbc_counts,
            "prediction_accuracy_model",
            "prediction_profit_model",
        )
        with open("assignment2_updated.json", "w") as f:
            json.dump(updated_json, f, indent=4)

        # Plot cumulative profit curve for Question 3.1
        plot_cumulative_profit(model, X_test)

        # Calculate costs for different strategies for Question 3.3
        cost_of_doing_nothing, cost_of_selling_all, cost_of_profit_model, savings = (
            calculate_costs(model, X_test, y_test)
        )
        print(
            "Cost of Doing Nothing: €{:.2f} million".format(
                cost_of_doing_nothing / 1000000
            )
        )
        print(
            "Cost of Selling All Claims: €{:.2f} million".format(
                cost_of_selling_all / 1000000
            )
        )
        print(
            "Cost When Using Profit Maximization Model: €{:.2f} million".format(
                cost_of_profit_model / 1000000
            )
        )
        print(
            "Estimated Savings to KindCorp: €{:.2f} million".format(savings / 1000000)
        )

        # Plot sensitivity analysis for Question 3.4
        plot_sensitivity_analysis(model, X_test, y_test)


if __name__ == "__main__":
    main()

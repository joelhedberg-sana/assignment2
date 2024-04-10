from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import json
import sqlalchemy
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# Constants and Configuration
MYSQL_HOST = "mysql-1.cda.hhs.se"
MYSQL_USERNAME = "be903"
MYSQL_PASSWORD = "robots"
MYSQL_SCHEMA = "Survivability"
CONNECTION_STRING = (
    f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_SCHEMA}"
)


def connect_to_database():
    """Establish a connection to the MySQL database."""
    return sqlalchemy.create_engine(CONNECTION_STRING)


def fetch_data(cnx):
    """Fetch data from the database."""
    exam_query = "SELECT patient_id, measurement, value FROM PatientExamination"
    patient_query = "SELECT * FROM Patient"
    study_query = "SELECT * FROM Study"

    exam_df = pd.read_sql(exam_query, cnx)
    patient_df = pd.read_sql(patient_query, cnx)
    study_df = pd.read_sql(study_query, cnx)

    return exam_df, patient_df, study_df


def process_wbc_data(exam_df):
    """Process White Blood Cell (WBC) data."""
    wbc_df = exam_df[exam_df["measurement"] == "White Blood Cell Count"].pivot(
        index="patient_id", columns="measurement", values="value"
    )
    if wbc_df.columns.nlevels > 1:
        wbc_df.columns = wbc_df.columns.droplevel()
    return wbc_df


def plot_wbc_distribution(wbc_df):
    """Plot the distribution of White Blood Cell counts."""
    plt.figure(figsize=(8, 5))
    plt.hist(
        wbc_df["White Blood Cell Count"].astype(float), bins=100, edgecolor="black"
    )
    plt.xlabel("White Blood Cell Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of White Blood Cell Counts")
    plt.xlim(0, 125)
    # plt.show()


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
        col for col in df.columns if col not in ["id", "patient_id", "died_in_hospital"]
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

    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model


def update_json_with_predictions(json_data, model, df):
    """Update the JSON data with the model's predictions."""
    # Prepare data for prediction
    features = [col for col in df.columns if col not in ['id', 'patient_id', 'died_in_hospital']]
    X = df[features]
    
    # One-hot encode categorical features
    X = pd.get_dummies(X)
    
    # Add missing (zero-valued) columns
    missing_cols = set(model.feature_names_in_) - set(X.columns)
    for c in missing_cols:
        X[c] = 0
    
    # Ensure the order of column in the test set is in the same order than in train set
    X = X[model.feature_names_in_]
    
    predictions = model.predict(X)
    
    # Update JSON data with predictions
    for i, patient in enumerate(json_data):
        patient["Prediction"] = int(predictions[i])


def main():
    cnx = connect_to_database()
    exam_df, patient_df, study_df = fetch_data(cnx)

    wbc_df = process_wbc_data(exam_df)
    plot_wbc_distribution(wbc_df)

    df = prepare_data_for_model(patient_df, exam_df, study_df)
    model = train_and_evaluate_model(df)

    with open("assignment2.json", "r") as f:
        json_data = json.load(f)

    update_json_with_predictions(json_data, model, df)


if __name__ == "__main__":
    main()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import json
import sqlalchemy
from datetime import datetime


# Connect to MySQL database
MYSQL_HOST = "mysql-1.cda.hhs.se"
MYSQL_USERNAME = "be903"
MYSQL_PASSWORD = "robots"
MYSQL_SCHEMA = "Survivability"
CONNECTION_STRING = (
    f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_SCHEMA}"
)
cnx = sqlalchemy.create_engine(CONNECTION_STRING)

# Fetch data from PatientExamination table
exam_query = "SELECT patient_id, measurement, value FROM PatientExamination"
exam_df = pd.read_sql(exam_query, cnx)

# Fetch data from Patient table
patient_query = "SELECT * FROM Patient"
patient_df = pd.read_sql(patient_query, cnx)

# Close database connection
cnx.dispose()
# Pivot exam_df to get white blood cell count per patient
wbc_df = exam_df[exam_df["measurement"] == "White Blood Cell Count"].pivot(
    index="patient_id", columns="measurement", values="value"
)
if wbc_df.columns.nlevels > 1:
    wbc_df.columns = wbc_df.columns.droplevel()  # drop redundant column level

# Plot distribution of white blood cell counts
plt.figure(figsize=(8, 5))
plt.hist(wbc_df["White Blood Cell Count"].astype(float), bins=100, edgecolor="black")
plt.xlabel("White Blood Cell Count")
plt.ylabel("Frequency")
plt.title("Distribution of White Blood Cell Counts")
plt.xlim(0, 125)
# plt.show()
# plt.savefig('wbc_distribution.png', dpi=200, bbox_inches='tight')

# Get classification set patient IDs (those with days_before_discharge = 0)
classification_ids = patient_df[patient_df["days_before_discharge"] > 0]["id"].tolist()

# Create list to store JSON data
json_data = []

# Loop through classification IDs
for patient_id in classification_ids:

    # Get rounded white blood cell count if exists, else None
    if patient_id in wbc_df.index:
        wbc_count = int(round(float(wbc_df.loc[patient_id, "White Blood Cell Count"])))
    else:
        wbc_count = None

    # Append to JSON data
    json_data.append(
        {
            "patient_id": patient_id,
            "prediction_accuracy_model": None,  # Placeholder for future data
            "white_blood_cell_count": wbc_count,
            "prediction_profit_model": None,  # Placeholder for future data
        }
    )

# Write JSON data to file
with open("assignment2.json", "w") as f:
    json.dump(json_data, f, indent=2)

# Drop unnecessary columns

# patient_df = patient_df.drop(['admission_date', 'days_before_discharge'], axis=1)


# Perform one-hot encoding on the 'language' column
language_dummies = pd.get_dummies(patient_df["language"], prefix="language")
patient_df = pd.concat([patient_df, language_dummies], axis=1)
patient_df = patient_df.drop("language", axis=1)

# Convert gender to numeric (0 for male, and 1 for female)
patient_df["gender"] = patient_df["gender"].map({"male": 0, "female": 1})

# Convert 'admission_date' to number of days since a reference date
reference_date = datetime(2017, 1, 1)
# Convert 'admission_date' to datetime
patient_df["admission_date"] = pd.to_datetime(patient_df["admission_date"])

# Now perform the subtraction
patient_df["admission_date"] = (patient_df["admission_date"] - reference_date).dt.days

# Replace non-finite values in 'do_not_resuscitate' with -1 and convert to integer
patient_df["do_not_resuscitate"] = (
    patient_df["do_not_resuscitate"].fillna(-1).astype(int)
)

# Split data into features (X) and target (y)
X = patient_df.drop(
    [
        "id",
        "doctors_2_months_survival_prediction",
        "doctors_6_months_survival_prediction",
    ],
    axis=1,
)
y = (
    patient_df["doctors_2_months_survival_prediction"] < 0.5
)  # Assuming doctors' 2-month prediction < 0.5 means the patient died

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Perform one-hot encoding on categorical columns
for col in cat_cols:
    dummies = pd.get_dummies(X[col], prefix=col)
    X = pd.concat([X, dummies], axis=1)
    X = X.drop(col, axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
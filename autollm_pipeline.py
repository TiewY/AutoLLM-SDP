import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib, csv, json, warnings
warnings.filterwarnings("ignore")

# ======== 1. Load Dataset ========
data = pd.read_csv("C:/Users/YingYing/Documents/FurtherStudy/UM/Sem 3 - Dissertation/AutoLLM/data/cm1.csv")

# ======== 2. Prepare Features and Target ========
X = data.drop('defects', axis=1)
y = data['defects'].apply(lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0)

# ======== 3. Normalize Features ========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======== 4. Split Train/Test ========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======== 5. AutoML with FLAML ========
automl = AutoML()
settings = {
    "time_budget": 300,
    "metric": "f1",
    "task": "classification",
    "log_file_name": "flaml_cm1.log"
}
automl.fit(X_train=X_train, y_train=y_train, **settings)

# ======== 6. Evaluate on Test Data ========
y_pred = automl.predict(X_test)

# 👉 HERE: print results directly to console
print("\n===== AutoML Results Summary =====")
print("Best model type:", automl.best_estimator)
print("Best hyperparameters:", automl.best_config)
print("Best validation loss:", automl.best_loss)
print("Training time (sec):", automl.best_config_train_time)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ======== 7. Save Model ========
joblib.dump(automl.model,
    "C:/Users/YingYing/Documents/FurtherStudy/UM/Sem 3 - Dissertation/AutoLLM/models/cm1_best_model.pkl")
print("✅ Model saved successfully!")

# ======== 8. Save Metadata ========
metadata = {
    "dataset": "CM1",
    "best_estimator": automl.best_estimator,
    "best_config": automl.best_config,
    "best_loss": automl.best_loss,
    "train_time": automl.best_config_train_time,
    "f1": f1_score(y_test, y_pred),
    "mcc": matthews_corrcoef(y_test, y_pred),
    "accuracy": accuracy_score(y_test, y_pred)
}

with open("C:/Users/YingYing/Documents/FurtherStudy/UM/Sem 3 - Dissertation/AutoLLM/models/cm1_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("✅ Metadata saved!")

# ======== 9. Append Results to CSV ========
results_path = "C:/Users/YingYing/Documents/FurtherStudy/UM/Sem 3 - Dissertation/AutoLLM/results/automl_results.csv"
header = ["Dataset", "BestModel", "F1", "MCC", "Accuracy", "TrainTime"]

try:
    with open(results_path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([metadata["dataset"], metadata["best_estimator"], metadata["f1"],
                         metadata["mcc"], metadata["accuracy"], metadata["train_time"]])
except FileExistsError:
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([metadata["dataset"], metadata["best_estimator"], metadata["f1"],
                         metadata["mcc"], metadata["accuracy"], metadata["train_time"]])

print("✅ Results saved to:", results_path)

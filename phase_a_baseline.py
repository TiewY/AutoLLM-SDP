import pandas as pd
from flaml import AutoML
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import csv
import json
import warnings
warnings.filterwarnings("ignore")

# ============================
# Utility: Find target column
# ============================
TARGET_COLS = ["defects", "bug", "isDefective", "label", "Defective"]

def detect_target_column(df):
    for col in TARGET_COLS:
        if col in df.columns:
            return col
    raise ValueError("❌ No valid defect/bug column found in dataset!")


# ============================
# Main Baseline Pipeline
# ============================
def run_automl_baseline(dataset_name, path):
    print(f"\n==============================")
    print(f">>> RUNNING BASELINE FOR {dataset_name}")
    print(f"==============================")

    # 1. Load dataset
    df = pd.read_csv(path)

    # 2. Detect target column
    target_col = detect_target_column(df)
    print(f"Detected target column: {target_col}")

    # 3. Prepare features & labels
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Convert string labels to 0/1
    y = y.apply(lambda x: 1 if str(x).strip().upper() in ["TRUE", "1", "YES"] else 0)

    # 4. Train/test split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================
    # 🔥 ADD 1: Handle imbalance (SMOTE)
    # ============================
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ============================
    # 🔥 ADD 2: Feature Selection
    # ============================
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(score_func=f_classif, k=20)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # ============================
    # 🔥 ADD 3: Normalize AFTER selection
    # ============================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. Run AutoML (FLAML)
    automl = AutoML()
    settings = {
        "time_budget": 600,
        "metric": "f1",
        "task": "classification",
        "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree"], 
        "log_file_name": f"logs/flaml_{dataset_name}.log",
    }
    automl.fit(X_train=X_train, y_train=y_train, **settings)

    # 7. Predictions & metrics
    y_pred = automl.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    result = {
        "dataset": dataset_name,
        "best_model": automl.best_estimator,
        "f1": f1,
        "mcc": mcc,
        "accuracy": acc,
        "train_time": automl.best_config_train_time
    }

    print("\n===== BASELINE RESULTS =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    # 8. Save model
    joblib.dump(
        automl.model,
        f"models/{dataset_name}_best_model.pkl"
    )

    # 9. Save metadata
    with open(f"models/{dataset_name}_metadata.json", "w") as f:
        json.dump(result, f, indent=4)

    # 10. Append baseline CSV
    csv_path = "results/baseline_results.csv"
    header = ["Dataset", "BestModel", "F1", "MCC", "Accuracy", "TrainTime"]

    try:
        with open(csv_path, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write header once
    except FileExistsError:
        pass  # File exists → append only

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            result["dataset"],
            result["best_model"],
            result["f1"],
            result["mcc"],
            result["accuracy"],
            result["train_time"]
        ])

    print(f"✅ Completed {dataset_name}. Results saved.\n")


# ============================
# Run Step A1, A2, A3
# ============================
if __name__ == "__main__":
    run_automl_baseline("CM1", "data/cm1.csv")
    run_automl_baseline("KC1", "data/kc1.csv")
    run_automl_baseline("PC1", "data/pc1.csv")

    print("\n🎉 Phase A (Baseline Experiments) Completed Successfully!")

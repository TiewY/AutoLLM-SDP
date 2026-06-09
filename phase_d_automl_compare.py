import os
import numpy as np
import json
import csv
import joblib

from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split

# Models
from flaml import AutoML
# from tpot import TPOTClassifier
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================
# Setup folders
# ============================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================
# Load embeddings (IMPORTANT)
# ============================
print("📂 Loading CodeT5 embeddings...")

train_emb = np.load("embeddings/train_codet5_final.npy")
test_emb = np.load("embeddings/test_codet5_final.npy")
train_labels = np.load("embeddings/train_labels_final.npy")
test_labels = np.load("embeddings/test_labels_final.npy")

# ============================
# Evaluation function
# ============================
def evaluate(name, y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n===== {name} RESULTS =====")
    print("F1:", f1)
    print("MCC:", mcc)
    print("Accuracy:", acc)

    return {
        "method": name,
        "f1": f1,
        "mcc": mcc,
        "accuracy": acc
    }

results = []
print("train_emb shape:", train_emb.shape)
print("train_labels shape:", train_labels.shape)

# ============================
# D1 — FLAML (Baseline)
# ============================
print("\n🚀 Running FLAML...")

flaml = AutoML()
flaml.fit(
    X_train=train_emb,
    y_train=train_labels,
    time_budget=1200,
    task="classification",
    metric="f1",
    estimator_list=["lgbm", "xgboost", "rf", "extra_tree"]
)

y_pred = flaml.predict(test_emb)
res = evaluate("FLAML", test_labels, y_pred)
results.append(res)

joblib.dump(flaml.model, "models/flaml_model.pkl")

# ============================
# D2 — TPOT
# ============================
# print("\n🚀 Running TPOT...")

# tpot = TPOTClassifier(
#     generations=5,
#     population_size=20,
#     verbosity=2,
#     random_state=42,
#     n_jobs=-1
# )

# tpot.fit(train_emb, train_labels)

# y_pred = tpot.predict(test_emb)
# res = evaluate("TPOT", test_labels, y_pred)
# results.append(res)

# tpot.export("models/tpot_pipeline.py")

# ============================
# D3 — Auto-sklearn (OPTIONAL - heavy)
# ============================
try:
    print("\n🚀 Running Auto-sklearn...")

    import autosklearn.classification

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=600,
        per_run_time_limit=60
    )

    automl.fit(train_emb, train_labels)

    y_pred = automl.predict(test_emb)
    res = evaluate("AutoSklearn", test_labels, y_pred)
    results.append(res)

except Exception as e:
    print("⚠️ Auto-sklearn skipped:", e)

# ============================
# D4 — Optuna (Hyperparameter tuning)
# ============================
print("\n🚀 Running Optuna...")

def objective(trial):
    model_type = trial.suggest_categorical("model", ["rf", "lr", "xgb", "lgbm"])

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15)
        )

    elif model_type == "lr":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10),
            max_iter=1000
        )

    elif model_type == "xgb":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("lr", 0.01, 0.3)
        )

    elif model_type == "lgbm":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            learning_rate=trial.suggest_float("lr", 0.01, 0.3)
        )

    from sklearn.model_selection import cross_val_score

    score = cross_val_score(
        model,
        train_emb,
        train_labels,
        cv=3,
        scoring="f1"
    ).mean()

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best Optuna Params:", best_params)

# Train final model
if best_params["model"] == "rf":
    best_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"]
    )
else:
    best_model = LogisticRegression(
        C=best_params["C"],
        max_iter=1000
    )

best_model.fit(train_emb, train_labels)
y_pred = best_model.predict(test_emb)

res = evaluate("Optuna", test_labels, y_pred)
results.append(res)

joblib.dump(best_model, "models/optuna_model.pkl")

# ============================
# Save Results
# ============================
csv_path = "results/phase_d_results.csv"

header = ["Method", "F1", "MCC", "Accuracy"]

try:
    with open(csv_path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
except:
    pass

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    for r in results:
        writer.writerow([
            r["method"],
            r["f1"],
            r["mcc"],
            r["accuracy"]
        ])

print("\n🎉 Phase D Completed!")
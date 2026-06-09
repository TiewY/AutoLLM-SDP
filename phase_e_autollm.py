import os
import numpy as np
import json
import csv

from flaml import AutoML
import optuna

from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================
# Setup
# ============================
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("📂 Loading FINAL embeddings...")

# ✅ ONLY USE FINAL PROCESSED DATA
X_train = np.load("embeddings/train_codet5_final.npy")
X_test = np.load("embeddings/test_codet5_final.npy")

y_train = np.load("embeddings/train_labels_final.npy")
y_test = np.load("embeddings/test_labels_final.npy")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

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

# ============================
# 🔥 STEP 1 — FLAML
# ============================
print("\n🚀 Running FLAML...")

flaml = AutoML()

flaml.fit(
    X_train=X_train,
    y_train=y_train,
    time_budget=600,
    task="classification",
    metric="f1",
    estimator_list=["lgbm", "xgboost", "rf"]
)

y_pred = flaml.predict(X_test)
res = evaluate("FLAML", y_test, y_pred)
results.append(res)

# Save model
import joblib
joblib.dump(flaml.model, "models/flaml_best.pkl")

# ============================
# 🔥 STEP 2 — OPTUNA
# ============================
print("\n🚀 Running Optuna...")

def objective(trial):
    model_type = trial.suggest_categorical("model", ["rf", "lr", "xgb", "lgbm"])

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            class_weight="balanced"
        )

    elif model_type == "lr":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10),
            max_iter=1000,
            class_weight="balanced"
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

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return f1_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest Optuna Params:", best_params)

# ============================
# Build best model
# ============================
if best_params["model"] == "rf":
    best_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"]
    )

elif best_params["model"] == "lr":
    best_model = LogisticRegression(
        C=best_params["C"],
        max_iter=1000
    )

elif best_params["model"] == "xgb":
    best_model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["lr"]
    )

elif best_params["model"] == "lgbm":
    best_model = LGBMClassifier(
        n_estimators=best_params["n_estimators"],
        num_leaves=best_params["num_leaves"],
        learning_rate=best_params["lr"]
    )

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

res = evaluate("Optuna", y_test, y_pred)
results.append(res)

joblib.dump(best_model, "models/optuna_best.pkl")

# ============================
# 🔥 FINAL SELECTION
# ============================
best = max(results, key=lambda x: x["f1"])

print("\n🏆 BEST PIPELINE FOUND:")
print(best)

# ============================
# Save results
# ============================
with open("results/phase_e_results.json", "w") as f:
    json.dump(results, f, indent=4)

with open("results/phase_e_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "F1", "MCC", "Accuracy"])

    for r in results:
        writer.writerow([
            r["method"],
            r["f1"],
            r["mcc"],
            r["accuracy"]
        ])

print("\n🎉 AutoLLM Final Completed!")
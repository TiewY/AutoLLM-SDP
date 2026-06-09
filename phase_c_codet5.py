import json
import os
import numpy as np
import torch
from tqdm import tqdm
from flaml import AutoML
from transformers import AutoTokenizer, T5EncoderModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib, csv

# ============================
# Setup folders
# ============================
os.makedirs("embeddings", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================
# Load dataset
# ============================
def load_codexglue(path):
    codes, labels = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            codes.append(item["func"])
            labels.append(item["target"])
    return codes, labels

train_codes, train_labels = load_codexglue("data/codexglue/train.jsonl")
test_codes, test_labels = load_codexglue("data/codexglue/test.jsonl")

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print("Train:", len(train_codes))
print("Test:", len(test_codes))

# ============================
# Embedding (CodeT5)
# ============================
train_emb_path = "embeddings/train_codet5.npy"
test_emb_path = "embeddings/test_codet5.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists("embeddings/train_codet5_final.npy"):
    print("✅ Loading FINAL embeddings...")
    train_emb = np.load("embeddings/train_codet5_final.npy")
    test_emb = np.load("embeddings/test_codet5_final.npy")
    train_labels = np.load("embeddings/train_labels_final.npy")
    test_labels = np.load("embeddings/test_labels_final.npy")

else:
    print("🚀 Generating CodeT5 embeddings...")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = T5EncoderModel.from_pretrained("Salesforce/codet5-base").to(device)
    model.eval()

    def get_embeddings_batch(texts, batch_size=16):
        vectors = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512   # 🔥 improved
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)

                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1)

                mean_pool = summed / counts
                vectors.extend(mean_pool.cpu().numpy())

        return np.array(vectors)

    train_emb = get_embeddings_batch(train_codes)
    test_emb = get_embeddings_batch(test_codes)

# ============================
# 🔥 Normalize embeddings
# ============================
# print("\n⚙️ Normalizing embeddings...")
# scaler = StandardScaler()
# train_emb = scaler.fit_transform(train_emb)
# test_emb = scaler.transform(test_emb)

# ============================
# 🔥 PCA (reduce noise)
# ============================
# print("\n⚙️ Applying PCA...")
# pca = PCA(n_components=256)
# train_emb = pca.fit_transform(train_emb)
# test_emb = pca.transform(test_emb)

# ============================
# 🔥 SMOTE (handle imbalance)
# ============================
# print("\n⚙️ Applying SMOTE...")
# print("Before:", np.bincount(train_labels))

# smote = SMOTE(random_state=42)
# train_emb, train_labels = smote.fit_resample(train_emb, train_labels)

np.save("embeddings/train_codet5_final.npy", train_emb)
np.save("embeddings/test_codet5_final.npy", test_emb)
np.save("embeddings/train_labels_final.npy", train_labels)
np.save("embeddings/test_labels_final.npy", test_labels)

print("After:", np.bincount(train_labels))

# ============================
# AutoML (stronger config)
# ============================
print("\n🚀 Running AutoML...")

automl = AutoML()
settings = {
    "time_budget": 1200,
    "metric": "f1",
    "task": "classification",
    "log_file_name": "logs/flaml_codet5.log",
    "estimator_list": ["lgbm", "rf", "xgboost", "extra_tree"]
}

automl.fit(X_train=train_emb, y_train=train_labels, **settings)

# ============================
# Evaluation
# ============================
y_pred = automl.predict(test_emb)

f1 = f1_score(test_labels, y_pred)
mcc = matthews_corrcoef(test_labels, y_pred)
acc = accuracy_score(test_labels, y_pred)

print("\n===== IMPROVED CODET5 RESULTS =====")
print("Best Model:", automl.best_estimator)
print("F1:", f1)
print("MCC:", mcc)
print("Accuracy:", acc)

# ============================
# Save model + results
# ============================
joblib.dump(automl.model, "models/codet5_model.pkl")

results = {
    "embedding": "CodeT5 (Improved)",
    "best_model": automl.best_estimator,
    "f1": f1,
    "mcc": mcc,
    "accuracy": acc
}

with open("models/codet5_metadata.json", "w") as f:
    json.dump(results, f, indent=4)

csv_path = "results/phase_c_results.csv"
header = ["Embedding", "BestModel", "F1", "MCC", "Accuracy"]

try:
    with open(csv_path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
except:
    pass

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(results.values())

print("\n🎉 Phase C Improved Completed!")
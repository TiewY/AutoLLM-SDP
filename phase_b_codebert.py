import json
import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from flaml import AutoML
from imblearn.over_sampling import SMOTE
import joblib
import csv

# ============================
# Setup folders
# ============================
os.makedirs("embeddings", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================
# Load CodeXGLUE dataset
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

print("Train samples:", len(train_codes))
print("Test samples:", len(test_codes))

# ============================
# Load CodeBERT
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()

# ============================
# 🔥 Improved Embedding (Batch + Mean Pooling)
# ============================
def get_embeddings_batch(texts, batch_size=16):
    vectors = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512   # 🔥 increased
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)

            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1)

            mean_pool = summed / counts
            vectors.append(mean_pool.cpu().numpy())

    return np.vstack(vectors)

# ============================
# Generate embeddings
# ============================
print("\n🚀 Generating CodeBERT embeddings...")
train_emb = get_embeddings_batch(train_codes)
test_emb = get_embeddings_batch(test_codes)

# ============================
# 🔥 Normalize embeddings
# ============================
print("\n⚙️ Normalizing embeddings...")
scaler = StandardScaler()
train_emb = scaler.fit_transform(train_emb)
test_emb = scaler.transform(test_emb)

# ============================
# 🔥 PCA (reduce noise)
# ============================
print("\n⚙️ Applying PCA...")
pca = PCA(n_components=256)
train_emb = pca.fit_transform(train_emb)
test_emb = pca.transform(test_emb)

# ============================
# 🔥 Handle imbalance (SMOTE)
# ============================
print("\n⚙️ Applying SMOTE...")
print("Before SMOTE:", np.bincount(train_labels))

smote = SMOTE(random_state=42)
train_emb, train_labels = smote.fit_resample(train_emb, train_labels)

print("After SMOTE:", np.bincount(train_labels))

# ============================
# Save embeddings (optional reuse)
# ============================
np.save("embeddings/train_codebert.npy", train_emb)
np.save("embeddings/test_codebert.npy", test_emb)
np.save("embeddings/train_labels.npy", train_labels)
np.save("embeddings/test_labels.npy", test_labels)

# ============================
# AutoML (FLAML)
# ============================
print("\n🚀 Running AutoML (FLAML)...")

automl = AutoML()
settings = {
    "time_budget": 1200,  # 🔥 increased
    "metric": "f1",
    "task": "classification",
    "log_file_name": "logs/flaml_codebert.log",
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

print("\n===== IMPROVED CODEBERT RESULTS =====")
print("Best Model:", automl.best_estimator)
print("F1:", f1)
print("MCC:", mcc)
print("Accuracy:", acc)

# ============================
# Save model
# ============================
joblib.dump(automl.model, "models/codebert_best_model.pkl")

# ============================
# Save results
# ============================
results = {
    "embedding": "CodeBERT (Improved)",
    "best_model": automl.best_estimator,
    "f1": f1,
    "mcc": mcc,
    "accuracy": acc
}

with open("models/codebert_metadata.json", "w") as f:
    json.dump(results, f, indent=4)

# Save CSV
csv_path = "results/phase_b_results.csv"
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

print("\n🎉 Phase B Improved Completed!")
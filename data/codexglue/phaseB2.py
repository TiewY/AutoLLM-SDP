from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)

def get_embeddings(texts):
    vectors = []
    for code in tqdm(texts):
        inputs = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        vectors.append(cls_vec[0])
    return np.array(vectors)

# Generate embeddings
train_emb = get_embeddings(train_codes)
test_emb = get_embeddings(test_codes)

# Save to disk
np.save("embeddings/train_codebert.npy", train_emb)
np.save("embeddings/test_codebert.npy", test_emb)
np.save("embeddings/train_labels.npy", train_labels)
np.save("embeddings/test_labels.npy", test_labels)
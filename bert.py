import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
print(torch.cuda.get_device_name(0))
# 1) Read Data (Only kept Positive and Negative)
df = pd.read_csv("googleplaystore_user_reviews.csv", usecols=["Translated_Review","Sentiment"]).dropna()
df["Sentiment"] = df["Sentiment"].str.strip().str.lower()
df = df[df["Sentiment"].isin(["positive","negative"])]
#0/1：positive=1, negative=0
y = (df["Sentiment"] == "positive").astype(int).to_numpy()
texts = df["Translated_Review"].astype(str).tolist()
# 2) Split Training and Testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)
# 3) Use Fine-Tuned Bert sentiment model
device = 0 if torch.cuda.is_available() else -1
clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# 4) Prediction for test set
preds = clf(X_test, batch_size=32, truncation=True)


# 5) Convert POSITIVE/NEGATIVE into 0/1
def to_num(lbl: str) -> int:
    s = lbl.upper()
    return 1 if s.startswith("POS") else 0


y_pred = np.array([to_num(p["label"]) for p in preds])

# 6) Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["negative","positive"]))

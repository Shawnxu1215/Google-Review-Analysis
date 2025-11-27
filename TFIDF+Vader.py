from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('googleplaystore.csv')
review_df = pd.read_csv('googleplaystore_user_reviews.csv')
# 1. Data Preparation
review_df = review_df[review_df['Sentiment'] != 'Neutral']
df = review_df[['Translated_Review','Sentiment']].dropna()
texts = df['Translated_Review'].tolist()
labels = df['Sentiment'].tolist()

# 2. TF‑IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(texts)      # (n_samples, 5000)

# 3. VADER
analyzer = SentimentIntensityAnalyzer()
vader_feats = np.array([
    [
      analyzer.polarity_scores(t)['neg']
    ]
    for t in texts
])                                        # (n_samples, 4)
vader_feats = StandardScaler().fit_transform(vader_feats)

# 4. Split train and test dataset
X_tfidf_tr, X_tfidf_te, vader_tr, vader_te, y_tr, y_te = train_test_split(
    X_tfidf, vader_feats, labels, test_size=0.2, random_state=42
)

# 5.Feature Matrix → (n_samples, 5004)
X_train = hstack([X_tfidf_tr, vader_tr])
X_test  = hstack([X_tfidf_te, vader_te])
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 6. Encoding
le = LabelEncoder()
y_train = le.fit_transform(y_tr)
y_test  = le.transform(y_te)

# 7. Training & Forecasting
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from transformers import pipeline
# Drop NA and Neutral Column
review_df = pd.read_csv('googleplaystore_user_reviews.csv')
review_df = review_df[['Translated_Review', 'Sentiment']].dropna()
review_df = review_df[review_df['Sentiment'] != 'Neutral']
print('After Dropping NA value and Neutral we have total sample size of ', len(review_df))
# Use Tfidf convert word to vectors and add models after that
tfidf = TfidfVectorizer(max_features=5000, min_df=10)
X = tfidf.fit_transform(review_df['Translated_Review'])
le = LabelEncoder()
y = le.fit_transform(review_df['Sentiment'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Result 1: TFIDF + Logistic Regression:')
print('Accuracy', accuracy_score(y_test, y_pred))
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
# desicion tree
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print('Result 2: TFIDF + decision tree classifier')
print('Accuracy', accuracy_score(y_test, y_pred2))
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred2))
print("Report:\n", classification_report(y_test, y_pred2))
# random forest
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print('Result 3: TFIDF + random forest classifier')
print('Accuracy', accuracy_score(y_test, y_pred3))
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred3))
print("Report:\n", classification_report(y_test, y_pred3))
# SVM
model4 = LinearSVC()
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print('Result 4: TFIDF + Support Vector Machine')
print('Accuracy', accuracy_score(y_test, y_pred4))
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred4))
print("Report:\n", classification_report(y_test, y_pred4))
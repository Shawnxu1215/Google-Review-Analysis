import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gensim.models import Word2Vec
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# Read Data
df = pd.read_csv('googleplaystore.csv')
review_df = pd.read_csv('googleplaystore_user_reviews.csv')
# Data Cleaning for Rating, Install, Price Column, Remove all Null/NA value
df = df[df['Rating'] <= 5]
df = df[df['Rating'].notnull()]
df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = pd.to_numeric(df['Price'])
# Make sure there is no NAN value
print('Total of Apps remaining(without NAN Value) ' + str(df['Price'].notnull().sum()))
print('Total count of sentiment polarity: ' + str(review_df['Sentiment_Polarity'].notnull().sum()))
sns.histplot(review_df['Sentiment_Polarity'].dropna(), bins=30, kde=True)
plt.title('Distribution of Sentiment Polarity')
plt.xlabel('Sentiment_Polarity')
plt.show()
Avg_Sentiment_Polarity = review_df.groupby('App')['Sentiment_Polarity'].mean().reset_index()
merged_df = pd.merge(df, Avg_Sentiment_Polarity, on='App', how='left')
Avg_Sentiment_Polarity.columns = ['App', 'Avg_Sentiment_Polarity']
# First correlation targeting paid Apps 
Paid_df = merged_df[merged_df['Price'] > 0]
print('Number of Paid App: ' + str(len(Paid_df)))
print('Correlation with Paid App:')
Cols_with_fee = ['Rating', 'Reviews', 'Installs', 'Price', 'Sentiment_Polarity']
corr_matrix = Paid_df[Cols_with_fee].corr()
print(corr_matrix)
# Second correlation targeting Free Apps
Free_df = merged_df[merged_df['Price'] == 0]
print('Number of Free App: ' + str(len(Free_df)))
print('Correlation with Free App:')
Cols_with_No_fee = ['Rating', 'Reviews', 'Installs', 'Sentiment_Polarity']
corr_matrix2 = Free_df[Cols_with_No_fee].corr()
print(corr_matrix2)
# multivariable regression for paid Apps
reg_df = Paid_df[Cols_with_fee]
x1 = reg_df[['Reviews', 'Installs', 'Price']]  # y ~ Reviews + Installs + Price
x1 = x1.astype(float)
y = reg_df['Rating']
x1 = sm.add_constant(x1)
model = sm.OLS(y, x1).fit()
print('First regression exclude Sentiment_Polarity For paid Apps')
print(model.summary())
reg_df2 = Paid_df[Cols_with_fee].dropna()
x2 = reg_df2[['Reviews', 'Installs', 'Price', 'Sentiment_Polarity']]  # y ~ Reviews + Installs + Price + Sentiment_Polarity
x2 = x2.astype(float)
x2 = sm.add_constant(x2)
y2 = reg_df2['Rating']
model2 = sm.OLS(y2, x2).fit()
print('Second regression include Sentiment_Polarity For Paid Apps')  # only have 15 Observation
print(model2.summary())
# multivariable regression for Free Apps
reg3_df = Free_df[Cols_with_No_fee]
x3 = reg3_df[['Reviews', 'Installs']]   # y ~ Reviews + Installs
x3 = x3.astype(float)
x3 = sm.add_constant(x3)
y3 = reg3_df['Rating']
model3 = sm.OLS(y3, x3).fit()
print('First regression exclude Sentiment_Polarity For Free Apps')
print(model3.summary())
reg4_df = Free_df[Cols_with_No_fee].dropna()
x4 = reg4_df[['Reviews', 'Installs', 'Sentiment_Polarity']] # y ~ Reviews + Installs + Sentiment_Polarity
x4 = x4.astype(float)
x4 = sm.add_constant(x4)
y4 = reg4_df['Rating']
model4 = sm.OLS(y4, x4).fit()
print('Second regression include Sentiment_Polarity For Free Apps')
print(model4.summary())
# Review Group by App's category, and get average App rating per category
Cols_to_groupby = ['Category', 'Rating', 'Sentiment_Polarity']
df2 = merged_df[Cols_to_groupby].dropna()
cols_to_convert = Cols_to_groupby[1:]  # 除了Category
df2[cols_to_convert] = df2[cols_to_convert].apply(pd.to_numeric, errors='coerce')
grouped_df = df2.groupby('Category').mean().reset_index()
print(grouped_df)
# Install for all Apps
x5 = merged_df[['Rating', 'Reviews', 'Price', 'Sentiment_Polarity']]
y5 = merged_df['Installs']
data5 = pd.concat([x5, y5], axis=1).dropna()
x5_clean = data5[['Rating', 'Reviews', 'Price', 'Sentiment_Polarity']].astype(float)
y5_clean = data5['Installs'].astype(float)
x5 = sm.add_constant(x5_clean)
model5 = sm.OLS(y5_clean, x5).fit()
print(model5.summary())
# Drop all NA, only analyze Rating and Installs
x = merged_df[['Rating']].dropna().astype(float)
y = merged_df.loc[x.index, 'Installs'].astype(float)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

# TFIDF + logistic regression
review_df = review_df[['Translated_Review', 'Sentiment']].dropna()
review_df = review_df[review_df['Sentiment'] != 'Neutral']
print(review_df)
tfidf = TfidfVectorizer(max_features=5000, min_df=10,)
X = tfidf.fit_transform(review_df['Translated_Review'])
le = LabelEncoder()
y = le.fit_transform(review_df['Sentiment'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Use transformer featuring into logistics regression
model2 = SentenceTransformer('all-MiniLM-L6-v2')
X = model2.encode(review_df['Translated_Review'].tolist())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, y_train)      
y_pred = model2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

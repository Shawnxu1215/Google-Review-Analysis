# App Review Sentiment Analysis & Rating Drivers

This repository contains the code for my app-review analytics project completed during my Data Scientist internship at **Alphabet Inc.**  

The goal of the project is to:

1. **Understand what drives app ratings** by combining app-level metadata with user reviews.
2. **Build and compare sentiment analysis models** (from rule-based baselines to BERT) on Google Play style review data.
3. **Provide interpretable outputs** that can support product and marketing decisions.

## Repository Structure

- **`Data Analysis + baseline.py`**  
  - Loads and cleans the raw reviews + metadata.  
  - Exploratory data analysis (distribution of ratings, review length, categories, etc.).  
  - Simple baseline models (e.g., majority class, bag-of-words + logistic regression).  
  - Correlation / regression to identify rating drivers (category, installs, price, sentiment, …).

- **`Machine Learning models.py`**  
  - Feature engineering for classical ML models (TF-IDF, n-grams, basic text cleaning).  
  - Training / evaluation of models such as:
    - Logistic Regression  
    - Linear SVM  
    - Random Forest / Gradient Boosting (depending on config)  
  - Cross-validation, hyperparameter tuning, and metrics (accuracy, precision/recall, F1, confusion matrix).

- **`TFIDF+Vader.py`**  
  - Rule-based / lexicon baseline using **VADER** sentiment scores.  
  - Combines VADER polarity scores with TF-IDF features.  
  - Compares lexicon-based sentiment to supervised ML models.

- **`bert.py`**  
  - End-to-end pipeline for fine-tuning a **BERT-style transformer** on review text.  
  - Tokenization, dataset/dataloader construction, training loop, and evaluation.  
  - Intended as the “upper-bound” model to compare against classical ML.

- **`.gitignore`**  
  - Standard ignores for Python virtual environments, cache files, and local data.

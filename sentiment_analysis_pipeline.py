import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Baseline Machine Learning Modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Advanced Deep Learning Modules
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

print("=========================================")
print("STARTING SENTIMENT ANALYSIS TASK PIPELINE")
print("=========================================\n")

# ---------------------------------------------------------
# STEP 1: PREPARE DATASET
# ---------------------------------------------------------
print("[Step 1] Creating a sample dataset...")

# Replace your current Step 1 dataset with this expanded one:
data = {
    "text": [
        "The company reported a 20% increase in net profit this quarter.",
        "Market shares plummeted following the unexpected resignation of the CEO.",
        "The stock price remained steady with no major fluctuations recorded.",
        "Excellent revenue growth and great future prospects for investors.",
        "Terrible performance, closing down several retail branches due to loss.",
        "Quarterly earnings met expectations with stable growth metrics.",
        "The regulatory body is launching a massive fraud investigation into the firm.",
        "Analysts upgraded the stock rating to a strong buy today.",
        "Net income dropped significantly compared to the previous fiscal year.",
        "The board announced a strategic partnership that will boost market value.",
        "Operating costs remained unchanged during the last quarter.",
        "The company faces a massive lawsuit that could lead to bankruptcy.",
        "Revenue increased by millions in a record-breaking sales streak.",
        "The factory closure resulted in hundreds of immediate job cuts."
    ],
    "label": [
        "positive", "negative", "neutral", "positive", "negative", 
        "neutral", "negative", "positive", "negative", "positive", 
        "neutral", "negative", "positive", "negative"
    ]
}
df = pd.DataFrame(data)
df.to_csv("sentiment_data.csv", index=False)
print("-> Dataset saved successfully as 'sentiment_data.csv'!\n")


# ---------------------------------------------------------
# STEP 2: BASELINE MODEL (TF-IDF + Logistic Regression)
# ---------------------------------------------------------
print("[Step 2] Training Baseline Model...")

# Load Data
df_loaded = pd.read_csv("sentiment_data.csv")

# Split data using the corrected 'test_size' parameter
X_train, X_test, y_train, y_test = train_test_split(
    df_loaded['text'], 
    df_loaded['label'], 
    test_size=0.25,      # Adjusted size slightly to fit small sample split
    random_state=42
)

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
baseline_model = LogisticRegression()
baseline_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = baseline_model.predict(X_test_tfidf)

# DELIVERABLE 1: Metrics Table Evaluation
print("\n--- BASELINE EVALUATION METRICS ---")
print(classification_report(y_test, y_pred, zero_division=0))

# DELIVERABLE 2: Save Confusion Matrix Image
cm = confusion_matrix(y_test, y_pred, labels=baseline_model.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=baseline_model.classes_, yticklabels=baseline_model.classes_)
plt.title('Baseline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

output_image_name = 'confusion_matrix.png'
plt.savefig(output_image_name)
print(f"-> Confusion matrix visualization saved to disk as '{output_image_name}'\n")
plt.close() # Close plot to free memory


# ---------------------------------------------------------
# STEP 3: ADVANCED PRE-TRAINED MODEL (FinBERT Inference)
# ---------------------------------------------------------
print("[Step 3] Initializing Advanced Pre-trained Model (FinBERT)...")

# Load lightweight, reliable FinBERT files
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Package into a simple sentiment pipeline
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print("\n--- ADVANCED MODEL INFERENCE RUN ---")
# Pick a test sentence to evaluate
sample_inference_text = "The firm experienced record-breaking revenue surge this fiscal period."
result = finbert_pipeline(sample_inference_text)

print(f"Input Text: \"{sample_inference_text}\"")
print(f"FinBERT Predicted Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.2%})")

print("\n=========================================")
print("TASK COMPLETED SUCCESSFULLY!")
print("=========================================")
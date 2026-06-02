# 📊 Financial Sentiment Analysis Pipeline

A two-stage NLP pipeline for financial text sentiment classification, combining a lightweight TF-IDF + Logistic Regression baseline with **FinBERT** — a domain-specific BERT model pre-trained on financial data.

---

## 🚀 Features

- Labeled financial dataset (14 samples across Positive / Negative / Neutral)
- Baseline model using TF-IDF vectorization + Logistic Regression
- Advanced inference using [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)
- Confusion matrix visualization saved as a `.png`
- Clean, step-by-step pipeline with printed progress logs

---

## 🗂️ Project Structure

```
.
├── sentiment_analysis.py     # Main pipeline script
├── sentiment_data.csv        # Auto-generated labeled dataset
├── confusion_matrix.png      # Auto-generated confusion matrix heatmap
└── README.md
```

---

## ⚙️ Requirements

Install dependencies via pip:

```bash
pip install pandas matplotlib seaborn scikit-learn transformers torch
```

> **Note:** `torch` is required by the `transformers` library for FinBERT inference. A CPU-only install is sufficient if no GPU is available.

---

## ▶️ Usage

Run the full pipeline with:

```bash
python sentiment_analysis.py
```

The script will automatically:
1. Generate and save the dataset as `sentiment_data.csv`
2. Train the TF-IDF + Logistic Regression baseline and print evaluation metrics
3. Save the confusion matrix as `confusion_matrix.png`
4. Download and run FinBERT inference on a sample sentence

---

## 🧪 Pipeline Overview

### Step 1 — Dataset Preparation

A small labeled dataset of 14 financial news sentences is created in-script covering three sentiment classes:

| Label    | Example |
|----------|---------|
| Positive | *"Analysts upgraded the stock rating to a strong buy today."* |
| Negative | *"The company faces a massive lawsuit that could lead to bankruptcy."* |
| Neutral  | *"Operating costs remained unchanged during the last quarter."* |

---

### Step 2 — Baseline Model (TF-IDF + Logistic Regression)

- Text is vectorized using `TfidfVectorizer` with English stop words removed
- A `LogisticRegression` classifier is trained on a 75/25 train-test split
- Outputs a `classification_report` (precision, recall, F1) and a confusion matrix heatmap

---

### Step 3 — Advanced Model (FinBERT)

- Loads `ProsusAI/finbert` from Hugging Face
- Runs inference via the `transformers` pipeline API
- Outputs the predicted sentiment label and confidence score for a sample sentence

**Example Output:**
```
Input Text: "The firm experienced record-breaking revenue surge this fiscal period."
FinBERT Predicted Sentiment: positive (Confidence: 99.81%)
```

---

## 📈 Sample Output

```
--- BASELINE EVALUATION METRICS ---
              precision    recall  f1-score   support
    negative       ...       ...       ...       ...
     neutral       ...       ...       ...       ...
    positive       ...       ...       ...       ...

--- ADVANCED MODEL INFERENCE RUN ---
FinBERT Predicted Sentiment: positive (Confidence: 99.81%)
```

> ⚠️ Baseline metrics may vary due to the small dataset size. This project is intended as a demonstrative NLP pipeline.

---

## 📦 Models Used

| Model | Type | Source |
|-------|------|--------|
| Logistic Regression | Classical ML | scikit-learn |
| FinBERT | Transformer (BERT-based) | [Hugging Face — ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) |

---

## 📝 License

This project is open for educational and research use. FinBERT is subject to its own [license terms](https://huggingface.co/ProsusAI/finbert).

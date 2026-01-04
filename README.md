# Fake News Detection Project

A simple machine learning project to detect fake news using text classification.

## Overview

This project uses Natural Language Processing and Machine Learning to classify news articles as fake or real.

## Dataset

- **Fake.csv**: Contains fake news articles
- **True.csv**: Contains real news articles
- **Labels**: Fake = 0, Real = 1

## Models Used

1. **Logistic Regression**: Linear classification model
2. **Naive Bayes**: Probabilistic classification model

## Features

- Text preprocessing (lowercase, remove punctuation, remove stopwords)
- TF-IDF vectorization for feature extraction
- 80/20 train-test split
- Model comparison and evaluation
- Custom prediction function

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Run Python Script

```bash
python fake_news_detection.py
```

### Run Jupyter Notebook

```bash
jupyter notebook Fake_News_Detection.ipynb
```

## Project Steps

1. Load datasets (Fake.csv and True.csv)
2. Add labels (Fake=0, Real=1)
3. Merge datasets
4. Preprocess text data
5. Convert text to TF-IDF features
6. Split data (80% train, 20% test)
7. Train Logistic Regression and Naive Bayes
8. Evaluate models
9. Compare and select best model
10. Make predictions on new articles

## Expected Results

- Accuracy: 85-95%
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## Files

- `fake_news_detection.py` - Main Python script
- `Fake_News_Detection.ipynb` - Jupyter notebook
- `requirements.txt` - Required libraries
- `README.md` - Project documentation
- `Fake.csv` - Fake news dataset
- `True.csv` - Real news dataset

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- nltk

## Notes

- Ensure CSV files are in the same directory
- The model trains on your specific dataset
- Some misclassifications are normal (no model is 100% accurate)
- Always verify important news from multiple sources

## Project Status

Ready for academic presentation and viva examination.

---

**Created for**: Machine Learning Course Project  
**Purpose**: Educational and Learning
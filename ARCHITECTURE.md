# System Architecture

## Overview
A multi-layer comment moderation pipeline designed to process YouTube comments 
and return a structured moderation recommendation. Built to function as a 
realistic deployment-ready pipeline, not a simple binary classifier.

## Pipeline
A comment enters the system and is processed sequentially through three layers 
before a recommendation is returned.

**Layer 1: Spam Filter**
The comment is first passed through a lightweight binary spam classifier 
(Logistic Regression + TF-IDF) trained on a combined YouTube spam dataset 
of 7,865 labeled comments. If flagged as spam the comment is immediately 
returned with a spam recommendation and does not proceed further.

**Layer 2: Toxicity Classifier**
Non-spam comments are passed to a fine-tuned RoBERTa model that performs 
multi-label classification across five categories: toxic, obscene, threat, 
insult, and identity_hate. Each category returns an independent probability 
score. The model was fine-tuned using focal loss with class weights to 
address label imbalance across 160,568 labeled comments.

**Layer 3: Confidence Arbitration**
The output probabilities are evaluated against confidence thresholds to 
determine the appropriate moderation action.

    confidence >= 0.60   auto_flag or auto_approve
    confidence 0.45-0.60  human_review
    confidence < 0.45    auto_approve

## Output
Every comment processed by the pipeline returns the following:

    {
        "comment": "original text",
        "spam": false,
        "labels": ["insult", "obscene"],
        "confidence": 0.76,
        "action": "human_review",
        "processing_time_ms": 42
    }

## Deployment
The application is deployed as a Streamlit app on Streamlit Community Cloud.
Models are hosted on HuggingFace Hub and loaded at runtime.

    App:    Streamlit Community Cloud
    Models: HuggingFace Hub (DavidMembreno)

## Project Structure

    youtube-comment-moderation/
    ├── data/
    │   ├── raw/
    │   │   ├── jigsaw/
    │   │   ├── spam/
    │   │   ├── spam_extra/
    │   │   └── toxicity/
    │   └── processed/
    │       ├── processed_toxicity.csv
    │       └── processed_spam.csv
    ├── models/
    │   └── spam_classifier.pkl
    ├── app/
    │   ├── app.py
    │   ├── pipeline.py
    │   └── requirements.txt
    ├── notebooks/
    │   ├── preprocessing.ipynb
    │   ├── modelselection.ipynb
    │   ├── train_spam.ipynb
    │   └── train_toxicity.ipynb
    ├── evaluate/
    ├── ARCHITECTURE.md
    ├── SETUP.md
    └── README.md

## Models
Layer 1 is a Logistic Regression classifier with TF-IDF vectorization 
(max_features=30000, ngram_range=(1,2), C=10, class_weight=balanced). 
Trained on 7,865 YouTube spam comments achieving 93% accuracy and 0.98 ROC-AUC.

Layer 2 is a fine-tuned RoBERTa model initialized from s-nlp/roberta_toxicity_classifier 
and fine-tuned for multi-label classification across five toxicity categories 
using focal loss and per-class weights over 6 epochs on an RTX 5070.
Final macro F1: 0.70, weighted F1: 0.79.

## Data
Toxicity: Jigsaw Toxic Comment Dataset + YouTube Toxicity Data (160,568 rows)
Spam: YouTube Spam Collection + supplementary spam datasets (7,865 rows)

Raw data is not committed to the repository. Processed datasets are 
available in data/processed/.
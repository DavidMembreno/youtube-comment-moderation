# System Architecture

## Overview
A multi-layer comment moderation pipeline designed to process YouTube comments 
and return a structured moderation recommendation. The system is built to 
function as a realistic deployment-ready pipeline, not a simple binary classifier.

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

    confidence >= 0.85   auto_flag or auto_approve
    confidence 0.50-0.85  human_review
    confidence < 0.50    escalate

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
The system is deployed as a decoupled full stack application.

    Backend:  FastAPI served on Railway
    Frontend: React application deployed on Vercel
    Models:   Hosted on HuggingFace Hub

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
    │   ├── spam_classifier.pkl
    │   └── roberta_toxicity/ (HuggingFace Hub)
    ├── app/
    │   ├── main.py (FastAPI)
    │   └── pipeline.py
    ├── notebooks/
    │   ├── preprocessing.ipynb
    │   ├── modelselection.ipynb
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
and fine-tuned for multi-label classification across five toxicity categories. 
Trained with focal loss and per-class weights to address label imbalance.

## Data
Training data is sourced from the following datasets and preprocessed into 
a unified schema before training.

Toxicity: Jigsaw Toxic Comment Dataset + YouTube Toxicity Data (160,568 rows)
Spam: YouTube Spam Collection + supplementary spam datasets (7,865 rows)

Raw data is not committed to the repository. Processed datasets are 
available in data/processed/.
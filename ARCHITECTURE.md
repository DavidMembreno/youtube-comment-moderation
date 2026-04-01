# System Architecture

## Overview
A multi-layer comment moderation pipeline designed to process YouTube comments 
and return a structured moderation recommendation. The system is built to 
function as a realistic deployment-ready pipeline, not a simple binary classifier.

## Pipeline
A comment enters the system and is processed sequentially through three layers 
before a recommendation is returned.

**Layer 1: Spam Filter**
The comment is first passed through a lightweight binary spam classifier trained 
on YouTube spam data. If flagged as spam the comment is immediately returned with 
a spam recommendation and does not proceed further.

**Layer 2: Toxicity Classifier**
Non-spam comments are passed to a fine-tuned RoBERTa model that performs 
multi-label classification across six categories: toxic, severe_toxic, obscene, 
threat, insult, and identity_hate. Each category returns an independent 
probability score.

**Layer 3: Confidence Arbitration**
The output probabilities are evaluated against confidence thresholds to determine 
the appropriate moderation action.

    confidence >= 0.85   auto_flag or auto_approve
    confidence 0.50-0.85  human_review
    confidence < 0.50    escalate

## Output
Every comment processed by the pipeline returns the following:

    {
        "comment": "original text",
        "spam": false,
        "labels": ["insult", "obscene"],
        "severity": "medium",
        "confidence": 0.76,
        "action": "human_review",
        "processing_time_ms": 42
    }

## Project Structure

    youtube-comment-moderation/
    ├── data/
    │   ├── raw/          
    │   │   ├── jigsaw/
    │   │   ├── spam/
    │   │   └── toxicity/
    │   └── processed/    
    ├── models/           
    ├── app/              
    ├── notebooks/        
    ├── evaluate/         
    ├── ARCHITECTURE.md   
    ├── SETUP.md          
    └── README.md

## Models
Layer 1 is a lightweight binary classifier trained on YouTube spam data.
Layer 2 is a fine-tuned RoBERTa model (s-nlp/roberta_toxicity_classifier) 
trained on a unified dataset combining Jigsaw Toxic Comments and YouTube 
toxicity data totaling 160,568 labeled comments across six categories.

## Data
Training data is sourced from three datasets and preprocessed into a unified 
schema before training. Raw data is not committed to the repository. Processed 
datasets are available in data/processed/.
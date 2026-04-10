# YouTube Comment Moderation System

A deployment-ready multi-layer comment moderation pipeline that processes 
YouTube comments and returns structured moderation recommendations. Built 
as a portfolio project to demonstrate applied ML, system design, and deployment.

## Live Demo
[Streamlit App — add URL here once deployed]

## What it does
A comment enters the pipeline and passes through three sequential layers. 
First a lightweight spam filter gates obvious spam before it reaches the 
heavier model. Non-spam comments are then classified by a fine-tuned RoBERTa 
model across five toxicity categories. A confidence arbitration layer 
determines the final moderation action — auto approve, human review, or auto flag.

The system supports single comment analysis and batch CSV upload, returning 
an enriched dataset with moderation labels and action flags ready for a 
human review queue.

## Stack
- App: Streamlit on Streamlit Community Cloud
- Models: Logistic Regression (spam) + fine-tuned RoBERTa (toxicity)
- Training: PyTorch + HuggingFace Transformers on RTX 5070
- Model hosting: HuggingFace Hub

## Model Performance
Layer 1 spam classifier: 93% accuracy, 0.98 ROC-AUC
Layer 2 toxicity classifier: 0.70 macro F1, 0.79 weighted F1

## Project Structure
See ARCHITECTURE.md for full pipeline design and project structure.

## Setup
See SETUP.md for environment setup and dataset download instructions.

## Data
Jigsaw Toxic Comment Dataset, YouTube Toxicity Data, and YouTube Spam 
Collection. Raw data is not committed to this repository. Processed datasets 
are available in data/processed/.

## Author
David Membreno — CLU Computer Science, May 2026
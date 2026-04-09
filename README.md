# YouTube Comment Moderation System

A deployment-ready multi-layer comment moderation pipeline that processes 
YouTube comments and returns structured moderation recommendations. Built 
as a portfolio project to demonstrate applied ML, system design, and 
full stack deployment.

## What it does
A comment enters the pipeline and passes through three sequential layers. 
First a lightweight spam filter gates obvious spam before it reaches the 
heavier model. Non-spam comments are then classified by a fine-tuned RoBERTa 
model across five toxicity categories. Finally a confidence arbitration layer 
determines the moderation action — auto approve, human review, or auto flag.

The system returns a structured JSON response per comment including predicted 
labels, confidence score, and recommended action. It also supports batch 
processing via CSV upload, returning an enriched dataset with moderation 
labels and action flags ready for a human review queue.

## Stack
- Backend: FastAPI on Railway
- Frontend: React on Vercel  
- Models: Logistic Regression (spam) + fine-tuned RoBERTa (toxicity)
- Training: PyTorch + HuggingFace Transformers on RTX 5070

## Models
Layer 1 spam classifier achieves 93% accuracy and 0.98 ROC-AUC on 7,865 
labeled YouTube comments.

Layer 2 toxicity classifier is fine-tuned from s-nlp/roberta_toxicity_classifier 
on 160,568 labeled comments across five categories using focal loss and 
class weighting to handle label imbalance.

## Project Structure
See ARCHITECTURE.md for full pipeline design and project structure.

## Setup
See SETUP.md for environment setup and dataset download instructions.

## Data
Jigsaw Toxic Comment Dataset, YouTube Toxicity Data, and YouTube Spam 
Collection. Raw data is not committed to this repository. Processed datasets 
are available in data/processed/.
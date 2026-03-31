# Project Setup Guide

## First Time Setup

### 1. Clone the repo
git clone git@github.com:DavidMembreno/youtube-comment-moderation.git
cd youtube-comment-moderation


### 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate


### 3. Install dependencies
pip install -r requirements.txt


### 4. Set up Kaggle API (one time only)
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json

Paste the following with your credentials:
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_TOKEN"}

Then lock the file permissions:
chmod 600 ~/.kaggle/kaggle.json


## Every Time You Open a New Terminal
source .venv/bin/activate


## Downloading Datasets

### Regular dataset
kaggle datasets download -d owner/dataset-name -p data/raw/

### Competition dataset
kaggle competitions download -c competition-name -p data/raw/

### Unzip into organized folder
unzip data/raw/dataset-name.zip -d data/raw/foldername/


## Project Structure
youtube-comment-moderation/
├── data/
│   ├── raw/          # Raw downloaded datasets, never edited
│   │   ├── jigsaw/
│   │   ├── spam/
│   │   └── toxicity/
│   └── load.py       # Dataset loading and exploration
├── models/           # Fine-tuning scripts and saved weights
├── app/              # Streamlit application
├── notebooks/        # EDA and experiments
├── evaluate/         # Benchmarking and classification reports
├── SETUP.md          # This file
└── README.md


## Notes
- Never commit model weights (*.pt, *.bin, *.safetensors) - they are in .gitignore
- Never commit raw CSVs - data/raw/ is in .gitignore
- Always work from project root, not subdirectories
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

## Running the App Locally
cd app
streamlit run app.py

## Downloading Datasets
Processed datasets are already available in data/processed/ and do not 
need to be re-downloaded. Raw datasets can be pulled from Kaggle if needed 
for reprocessing.

### Toxicity datasets
kaggle datasets download -d julian3833/jigsaw-toxic-comment-classification-challenge -p data/raw/
kaggle datasets download -d reihanenamdari/youtube-toxicity-data -p data/raw/

### Spam datasets
kaggle datasets download -d ahsenwaheed/youtube-comments-spam-dataset -p data/raw/spam/
kaggle datasets download -d madhuragl/5000-youtube-spamnot-spam-dataset -p data/raw/spam_extra/
kaggle datasets download -d prashant111/youtube-spam-collection -p data/raw/spam_extra/

### Unzip
unzip data/raw/filename.zip -d data/raw/foldername/

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

## Notes
- Never commit model weights (*.pt, *.bin, *.safetensors) — in .gitignore
- Raw zip files are ignored by git
- Processed datasets in data/processed/ are tracked and committed
- Always work from project root, not subdirectories
- Models are hosted on HuggingFace Hub, not committed to this repo
- Developed and trained on WSL2 with an RTX 5070
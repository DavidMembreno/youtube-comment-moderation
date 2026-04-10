import torch
import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import hf_hub_download
import time

LABEL_COLS = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load spam classifier
def load_spam_model():
    path = hf_hub_download(
        repo_id='DavidMembreno/youtube-comment-moderation-spam',
        filename='spam_classifier.pkl'
    )
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load RoBERTa
def load_toxicity_model():
    tokenizer = RobertaTokenizer.from_pretrained(
        'DavidMembreno/youtube-comment-moderation-roberta'
    )
    model = RobertaForSequenceClassification.from_pretrained(
        'DavidMembreno/youtube-comment-moderation-roberta'
    )
    model.eval()
    return tokenizer, model

def get_action(confidence, threshold):
    if confidence >= threshold + 0.15:
        return 'auto_flag'
    elif confidence >= threshold:
        return 'human_review'
    else:
        return 'auto_approve'

def predict(text, spam_model, tokenizer, roberta_model, threshold=0.45):
    start = time.time()

    # Layer 1 - spam
    spam_pred = spam_model.predict([text])[0]
    spam_prob = spam_model.predict_proba([text])[0][1]

    if spam_pred == 1:
        return {
            'comment': text,
            'spam': True,
            'spam_confidence': round(float(spam_prob), 3),
            'labels': [],
            'label_scores': {},
            'confidence': round(float(spam_prob), 3),
            'action': 'auto_flag',
            'processing_time_ms': round((time.time() - start) * 1000, 2)
        }

    # Layer 2 - toxicity
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=128, padding=True
    )
    with torch.no_grad():
        logits = roberta_model(**inputs).logits
        probs = torch.sigmoid(logits)[0].tolist()

    label_scores = {label: round(prob, 3) for label, prob in zip(LABEL_COLS, probs)}
    triggered = [label for label, prob in label_scores.items() if prob >= threshold]
    confidence = round(max(probs), 3)
    action = get_action(confidence, threshold)

    return {
        'comment': text,
        'spam': False,
        'spam_confidence': round(float(spam_prob), 3),
        'labels': triggered,
        'label_scores': label_scores,
        'confidence': confidence,
        'action': action,
        'processing_time_ms': round((time.time() - start) * 1000, 2)
    }
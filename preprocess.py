import re

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|https\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", ' ', text)
    text = ' '.join(text.split())
    text = text.strip()
    return text

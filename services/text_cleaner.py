import re

def clean_text(text):
    # Convert the text to lowercase (optional, you can skip this if needed)
    text = text.lower()
    # Remove all non-word characters (alphanumeric characters and underscores)
    text = re.sub(r'\W', ' ', text)
    # Remove all digits
    text = re.sub(r'\d', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
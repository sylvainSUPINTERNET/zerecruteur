from typing import List
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
import torch
import pinecone
import re
import uuid

pinecone.init(api_key="", environment="asia-southeast1-gcp-free")

# TODO must be done only one time
# pinecone.create_index("backend-engineer-index", dimension=768)


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

def save_to_pinecone(uid, embedding_list,idx_name):
    
    idx = pinecone.Index(idx_name)
    idx.upsert(items=[uid], vectors=[
            (
            f"{str(uuid.uuid4())}",                # Vector ID 
            embedding_list,  # Dense vector values
            {"genre": "cv_backend_engineer"}     # Vector metadata
            )
        ], namespace="backend-engineer-namespace")
    print("Inserted with success")

def main():
    global_text:List[str] = list()

    print("Loading PDF file...")
    reader = PdfReader("cv_1.pdf")    
    for page in reader.pages:
        global_text.append(page.extract_text())
        
    
    global_text = [clean_text(text) for text in global_text]
    
    print("Loading model...")

    # Charger le modèle pré-entraîné et le tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
        
    # Prétraitement du texte et encodage
    # Tokenize and truncate the input text if it exceeds the maximum length
    # max lenght is the dimension of pinecone index !
    inputs = tokenizer(global_text, max_length=768, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)

    # Récupérer le vecteur d'embedding
    embedding = outputs.last_hidden_state.mean(dim=1)  # Moyenne des embeddings des tokens

    # Convertir le vecteur d'embedding en une liste Python
    embedding_list = embedding.tolist()[0]
    print("Save embedding list...")
    save_to_pinecone(uuid.uuid4().__str__(), embedding_list, "backend-engineer-index")

if __name__ == "__main__":
    main()
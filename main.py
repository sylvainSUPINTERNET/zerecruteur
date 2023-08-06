from typing import Annotated, List
import uuid
import pinecone
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from constants.constants import INDEXE_CV, INDEXE_JOB_ANNOUNCES
from services.db import save_to_pinecone

from services.extract_text_from_pdf import extract_text_from_pdf
from services.ia import load_model_and_tokenizer
from dotenv import load_dotenv
import os

load_dotenv()

pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_API_ENV'))

# TODO : create index if not exist
# TODO : FREE TIER ONLY 1 POD KEKW ONLY 1 INDEXE
#  found_idx = list()
# for idx_name in pinecone.list_indexes():
#     found_idx.append(idx_name)
    
# if INDEXE_CV not in found_idx:
#     pinecone.create_index(INDEXE_CV, dimension=768)
# if INDEXE_JOB_ANNOUNCES not in found_idx:
#     pinecone.create_index(INDEXE_JOB_ANNOUNCES, dimension=768)


app = FastAPI()


origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/upload/cv/")
async def create_file(file: Annotated[bytes, File()]):
    
    global_text:List[str] = extract_text_from_pdf(file)
    model, tokenizer = load_model_and_tokenizer()
    
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
    save_to_pinecone(uuid.uuid4().__str__(), embedding_list, INDEXE_CV)
    
    return {"message": "upload with success"}

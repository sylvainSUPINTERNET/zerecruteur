from typing import Annotated, List

from fastapi import File
from PyPDF2 import PdfReader
from io import BytesIO

from services.text_cleaner import clean_text


def extract_text_from_pdf(file: Annotated[bytes, File()])->List[str]:
    
    print("loading pdf file ...")
    global_text:List[str] = list()
    reader = PdfReader(BytesIO(file))
    
    for page in reader.pages:
        global_text.append(page.extract_text())
        
    
    
    global_text = [clean_text(text) for text in global_text]
    
    return global_text
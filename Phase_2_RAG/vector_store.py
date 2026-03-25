import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

import warnings
warnings.filterwarnings("ignore")

def ingest_data(file_path="../Phase_1_Scraper/courses_data.json", persist_directory="chroma_db"):
    print("Loading data from JSON...")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Run scraper.py first.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("Chunking data...")
    # 500 characters chunk with 50 overlap ensures that small batches/course info stay together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\\nNEW\\n", "\\n\\n", "\\n", " ", ""]
    )
    
    documents = []
    for item in data:
        chunks = text_splitter.split_text(item["content"])
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": item["source"]}))
            
    print(f"Loading local HuggingFace embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Creating ChromaDB and indexing {len(documents)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    print(f"Vectorstore created and persisted successfully in '{persist_directory}' directory!")

if __name__ == "__main__":
    ingest_data()

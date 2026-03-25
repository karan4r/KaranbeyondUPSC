import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
# Load .env from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def ingest_data_pinecone(file_path="../Phase_1_Scraper/courses_data.json"):
    print("Loading data from JSON...")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Run Phase_1_Scraper/scraper.py first.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("Chunking data...")
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
            
    hf_token = os.getenv("HF_TOKEN")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "karan-beyond-upsc")

    if not hf_token or not pinecone_api_key:
        print("ERROR: Missing HF_TOKEN or PINECONE_API_KEY in environment or .env file.")
        print("Please add these keys to your .env file before running this script.")
        return

    print("Loading HuggingFace Inference API embeddings...")
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token, 
        repo_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"Uploading {len(documents)} chunks to Pinecone Index '{pinecone_index_name}'...")
    
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=pinecone_index_name,
        pinecone_api_key=pinecone_api_key
    )
    
    print("Vectorstore created and uploaded to Pinecone successfully!")

if __name__ == "__main__":
    ingest_data_pinecone()

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore")

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is the price of UPSC Prarambh?"
print(f"Querying: {query}")
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"\\nResult {i+1}:\\n{doc.page_content}")

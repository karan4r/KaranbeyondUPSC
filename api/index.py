import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

app = FastAPI(title="Karan beyond UPSC API", description="RAG Serverless API for PhysicsWallah Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    api_key: str = None

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    groq_api_key = request.api_key or os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "karan-beyond-upsc")
    hf_token = os.getenv("HF_TOKEN")

    if not pinecone_api_key or not hf_token:
        raise HTTPException(status_code=500, detail="Server misconfigured: PINECONE_API_KEY or HF_TOKEN is missing in Vercel ENV.")
        
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=hf_token, 
            repo_id="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        if not groq_api_key:
            docs = retriever.invoke(query)
            if not docs:
                return {"answer": "I don't have enough context in my database about this.", "fallback": True}
            
            formatted_docs = "\n\n---\n\n".join([f"**From Source:** {d.metadata.get('source', 'PW')}\n{d.page_content}" for d in docs])
            return {
                "answer": f"API Key not provided. Here is the raw course data found in the database:\n\n{formatted_docs}",
                "fallback": True
            }
        else:
            if "GROQ_API_KEY" not in os.environ and groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
                
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
            
            system_prompt = (
                "You are an elite AI assistant for a chatbot named 'Karan beyond UPSC'. "
                "You assist users in providing information specifically about courses offered by Physics Wallah. "
                "Use the following pieces of retrieved context to answer the user's question clearly and concisely. "
                "Format your answer using markdown, bullet points, and bold text to make it easy to read. "
                "If the database lacks specific details regarding an off-topic issue, say that you don't have enough information based on the scraped data. "
                "Always maintain a polite, encouraging, and helpful tone.\n\n"
                "Context:\n{context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}"),
            ])
            
            combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            result = retrieval_chain.invoke({"input": query})
            return {"answer": result["answer"], "fallback": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

app = FastAPI(title="Karan beyond UPSC API", description="RAG API for PhysicsWallah Chatbot")

# CORS Configuration for the frontend to access API over localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store heavily only once at startup
print("Initializing Embeddings and Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="../Phase_2_RAG/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

class QueryRequest(BaseModel):
    query: str
    api_key: str = None  # Optional API key for LLM wrapper (can be GROQ_API_KEY)

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    query = request.query
    # Use provided api_key or fall back to environment variable
    api_key = request.api_key or os.getenv("GROQ_API_KEY")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        # Fallback mechanism if no API key is provided
        if not api_key:
            docs = retriever.invoke(query)
            if not docs:
                return {"answer": "I don't have enough context in my database about this.", "fallback": True}
            
            # Format raw docs nicely using Markdown line breaks
            formatted_docs = "\\n\\n---\\n\\n".join([f"**From Source:** {d.metadata.get('source', 'PW')}\\n{d.page_content}" for d in docs])
            return {
                "answer": f"API Key not provided. Here is the raw course data found in the database:\\n\\n{formatted_docs}",
                "fallback": True
            }
            
        else:
            # RAG via Groq
            if "GROQ_API_KEY" not in os.environ and api_key:
                os.environ["GROQ_API_KEY"] = api_key
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
            
            system_prompt = (
                "You are an elite AI assistant for a chatbot named 'Karan beyond UPSC'. "
                "You assist users in providing information specifically about courses offered by Physics Wallah. "
                "Use the following pieces of retrieved context to answer the user's question clearly and concisely. "
                "Format your answer using markdown, bullet points, and bold text to make it easy to read. "
                "If the database lacks specific details regarding an off-topic issue, say that you don't have enough information based on the scraped data. "
                "Always maintain a polite, encouraging, and helpful tone.\\n\\n"
                "Context:\\n{context}"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

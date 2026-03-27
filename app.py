import os
import streamlit as st
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Karan beyond UPSC", page_icon="🤖", layout="centered")

def get_secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

HF_TOKEN = get_secret("HF_TOKEN")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME") or "karan-beyond-upsc"

with st.sidebar:
    st.title("Settings")
    st.write("Ensure your API keys are configured.")
    
    default_groq = get_secret("GROQ_API_KEY") or ""
    groq_api_key = st.text_input("Groq API Key (Optional)", type="password", value=default_groq, help="Leave blank if configured in system secrets.")
    
    if not HF_TOKEN or not PINECONE_API_KEY:
        st.error("⚠️ Missing HF_TOKEN or PINECONE_API_KEY in secrets.")

st.title("🤖 Karan beyond UPSC")
st.markdown("Your AI assistant for PhysicsWallah courses.")

@st.cache_resource
def init_vectorstore():
    if not HF_TOKEN or not PINECONE_API_KEY:
        return None
        
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, 
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore

vstore = init_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you find a course today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a course..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not vstore:
            st.error("System is misconfigured. Cannot access the vector database. Please configure Pinecone and HF tokens.")
            st.stop()
            
        retriever = vstore.as_retriever(search_kwargs={"k": 4})
        
        if not groq_api_key:
            docs = retriever.invoke(prompt)
            if not docs:
                response = "I don't have enough context in my database about this."
            else:
                formatted_docs = "\n\n---\n\n".join([f"**From Source:** {d.metadata.get('source', 'PW')}\n{d.page_content}" for d in docs])
                response = f"⚠️ *No Groq API Key provided. Here is the raw course data found in the database:*\n\n{formatted_docs}"
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            message_placeholder = st.empty()
            
            if "GROQ_API_KEY" not in os.environ:
                os.environ["GROQ_API_KEY"] = groq_api_key
            
            try:
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=groq_api_key)
                
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
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                    
                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                full_response = ""
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                import traceback
                error_str = str(e).strip()
                error_repr = repr(e)
                
                # Check for the dreaded Hugging Face cold start KeyError
                if error_str == "0" or error_str == "'0'" or "KeyError" in error_repr:
                    st.warning("⏳ **The Hugging Face Embedding Model is currently waking up.**\n\nSince we are using the free Serverless Inference API, the model goes to sleep after inactivity. It usually takes about 20-30 seconds to load into memory.\n\n**Please wait 30 seconds and try submitting your question again!**")
                else:
                    st.error(f"Error connecting to backend services: {error_str}")
                    with st.expander("Show detailed error logs"):
                        st.code(traceback.format_exc())
                
                # To prevent leaving a hanging typing indicator or empty message
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Encountered an error. (See logs for details)"})

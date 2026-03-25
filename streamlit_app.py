__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Karan beyond UPSC", page_icon="📚")
st.title("📚 Karan beyond UPSC")
st.write("Welcome! I am your AI assistant for Physics Wallah courses. Ask me anything about our batches!")

# Sidebar for Configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("This chatbot requires a GROQ API key to generate natural language responses. If you do not provide one, the chatbot will still return the raw retrieved course information from the database.")
api_key = st.sidebar.text_input("GROQ API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))

@st.cache_resource
def load_vectorstore():
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load ChromaDB from local disk - Note path changes for root directory deployment
    vectorstore = Chroma(persist_directory="./Phase_2_RAG/chroma_db", embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = load_vectorstore()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What courses do you have for UPSC?"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("No GROQ API Key provided. Returning retrieved document context instead of AI generated answer:")
            docs = retriever.invoke(prompt)
            if not docs:
                response = "Could not find related course information."
            else:
                response = "\n\n---\n\n".join([f"**Source:** {d.metadata.get('source', 'Unknown')}\n\n{d.page_content}" for d in docs])
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            if "GROQ_API_KEY" not in os.environ and api_key:
                os.environ["GROQ_API_KEY"] = api_key
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0) # Using fast Groq model
            
            system_prompt = (
                "You are an AI assistant for a chatbot named 'Karan beyond UPSC'. "
                "You assist users in providing information specifically about courses offered by Physics Wallah. "
                "Use the following pieces of retrieved context to answer the user's question clearly and concisely. "
                "If you don't know the answer, say that you don't have enough information based on the scraped data. "
                "Always maintain a polite and helpful tone.\n\n"
                "Context:\n{context}"
            )
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}"),
            ])
            
            combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            with st.spinner("Searching and generating answer..."):
                try:
                    result = retrieval_chain.invoke({"input": prompt})
                    response = result["answer"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

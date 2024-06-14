import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import tempfile
import mysql.connector
from mysql.connector import Error

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="The Custodian",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("The Custodian 📑")
st.markdown(
    """
    Welcome to the Custodian! Upload your PDF documents, ask questions, 
    and get answers based on the content of the documents.
    """
)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding(documents):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings
        st.write("Documents have been embedded and vector store is created.")

uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        loaded_docs = loader.load()
        documents.extend(loaded_docs)
        st.success(f"Successfully loaded {len(loaded_docs)} pages from {uploaded_file.name}.")
    
    if st.button("Create Vector Store"):
        vector_embedding(documents)
        st.success("Vector Store is ready")

st.header("Ask Questions")
prompt1 = st.text_input("Enter Your Question From Documents")

def save_to_database(query, answer):
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 3306),  # Default MySQL port is 3306
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = """INSERT INTO query_answers (User_query, Answer) VALUES (%s, %s)"""
            cursor.execute(insert_query, (query, answer))
            connection.commit()
            cursor.close()
            connection.close()
            st.success("Query and answer saved to database.")
    except Error as e:
        st.error(f"Error: {e}")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.success(f"Response time: {time.process_time() - start:.2f} seconds")
        answer = response['answer']
        st.header("Answer")
        st.write(answer)

        # Save query and answer to the database
        save_to_database(prompt1, answer)

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
       st.warning("Please upload and process documents first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Sahil Janak")
st.sidebar.markdown("[GitHub Repo](https://github.com/your-repo)")


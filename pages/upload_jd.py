import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

# Page Heading & Title
st.set_page_config(page_title="Resume Ranker")
st.title("Resume Ranker")
st.caption("Powered by Devansh Rajani")
st.write("## Please Upload your Job Description file")
uploaded_jd = st.file_uploader(label="",type=['pdf'])

# "Process JD" for vector embedding process
if st.button("Process JD"):
    if uploaded_jd is not None:
        with st.spinner("Processing JD Please Wait..."):
            
            # Extract content from JD
            jd_text = ""
            for page in PdfReader(uploaded_jd).pages:
                jd_text += page.extract_text()

            # Make chunks of JD
            jd_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(jd_text)

            # Make embedding vectors of jd_chunks
            embedding_model = AzureOpenAIEmbeddings(
                openai_api_key=os.getenv("AZURE_API_KEY"),
                azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_API_BASE"),
                azure_deployment=os.getenv("EMBEDDING_AZURE_OPENAI_API_NAME"),
                chunk_size=10
            )

            # Store embedding vectors using FAISS
            jd_vectors = FAISS.from_texts(jd_chunks, embedding_model)

            # Store jd_vectors in session state and Show success message
            st.session_state.jd_vectors = jd_vectors
            st.success("JD processed sucessfully!!")

    else:
        st.error("Please enter JD first!!")

# Check if jd_vectors stored in session state
if "jd_vectors" in st.session_state:
    st.page_link(r"C:\Users\Devansh\Desktop\DS_Projects\Resume_ranker\pages\upload_resume.py",label="Next page for Resume upload")

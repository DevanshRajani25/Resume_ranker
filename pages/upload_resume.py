import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Upload Resume")
st.title("Upload Resumes")
st.write("## Upload multiple candidate's Resumes")

res_upload = st.file_uploader(
    label="Choose PDF files", 
    type=['pdf'], 
    accept_multiple_files=True,
    key="resume_uploader"
)

# Check if JD vector exists
if "jd_vectors" not in st.session_state:
    st.warning("Please enter JD first!!")
    st.info("Go to the Job Description page first to upload and process the JD.")
else:
    # Show the button regardless of file upload status
    if st.button("Rank Resumes", disabled=(not res_upload)):
        if res_upload and len(res_upload) > 0:
            with st.spinner("Analyzing Resumes..."):
                try:
                    # Embedding model
                    embedding_model = AzureOpenAIEmbeddings(
                        openai_api_key=os.getenv("AZURE_API_KEY"),
                        azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_API_BASE"),
                        azure_deployment=os.getenv("EMBEDDING_AZURE_OPENAI_API_NAME"),
                        chunk_size=10
                    )

                    resume_scores = []
                    processed_count = 0

                    for res in res_upload:
                        try:
                            st.write(f"Processing: {res.name}")
                            
                            # Extract text from PDF
                            res_text = ""
                            pdf_reader = PdfReader(res)
                            
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    res_text += text + "\n"
                            
                            # Check if text was extracted
                            if res_text.strip() == "":
                                st.warning(f"Couldn't extract text from {res.name}, skipping.")
                                continue
                            
                            # Chunk resume text
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=500,
                                chunk_overlap=50
                            )
                            res_chunks = text_splitter.split_text(res_text)
                            
                            if not res_chunks:
                                st.warning(f"No valid chunks created from {res.name}, skipping.")
                                continue

                            # Create vector store for resume chunks
                            res_vectors = FAISS.from_texts(res_chunks, embedding_model)

                            # Compare resume chunks with jd vectors
                            top_matches = st.session_state.jd_vectors.similarity_search_with_score(
                                res_chunks[0], k=min(5, len(res_chunks))
                            )

                            # Calculate average similarity score 
                            if top_matches:
                                avg_score = sum([score for _, score in top_matches]) / len(top_matches)
                                resume_scores.append((res.name, avg_score, res_text[:200] + "..."))
                                processed_count += 1
                            else:
                                st.warning(f"No similarity matches found for {res.name}")

                        except Exception as e:
                            st.error(f"Error processing {res.name}: {str(e)}")
                            continue

                    if resume_scores:
                        # Sort by lowest average score (most similar in FAISS cosine distance)
                        ranked_resumes = sorted(resume_scores, key=lambda x: x[1])

                        st.success(f"Successfully processed and ranked {processed_count} resumes!")
                        st.write("### 📊 Resume Rankings")
                        st.write("*Lower score indicates better match with Job Description*")

                        # Display rankings 
                        for idx, (filename, score, preview) in enumerate(ranked_resumes, 1):
                            with st.expander(f"#{idx} - {filename} (Score: {score:.4f})"):
                                st.write("**Preview:**")
                                st.write(preview)
                                
                                if score < 0.5:
                                    st.success("🟢 Excellent Match")
                                elif score < 0.7:
                                    st.info("🟡 Good Match")
                                else:
                                    st.warning("🔴 Poor Match")

                        # Summary statistics
                        st.write("### 📈 Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Resumes", len(ranked_resumes))
                        with col2:
                            st.metric("Best Score", f"{min(score for _, score, _ in ranked_resumes):.4f}")
                        with col3:
                            st.metric("Average Score", f"{sum(score for _, score, _ in ranked_resumes) / len(ranked_resumes):.4f}")

                    else:
                        st.error("No resumes could be processed successfully. Please check your PDF files.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.write("Please check your Azure OpenAI configuration and try again.")

        else:
            st.error("Please upload at least one resume file.")

# Display current status
if res_upload:
    st.write(f"**Files uploaded:** {len(res_upload)}")
    for file in res_upload:
        st.write(f"- {file.name} ({file.size} bytes)")
else:
    st.info("No files uploaded yet. Please select PDF files to rank.")
# 🧠 AI Resume Ranker <br/>

An AI-powered tool that ranks multiple resumes based on a given Job Description (JD) using semantic similarity.<br/>

## 🚀 Features<br/>

- Upload JD (PDF or text)<br/>
- Upload multiple resumes (PDF/DOCX)<br/>
- Extract & embed text using HuggingFace model (`all-MiniLM-L6-v2`)<br/>
- Vector similarity via FAISS<br/>
- Ranks resumes based on JD relevance<br/>

## 🛠️ Tech Stack<br/>

- Streamlit (frontend)<br/>
- PyMuPDF, python-docx (parsing)<br/>
- sentence-transformers (embeddings)<br/>
- FAISS via LangChain (similarity search)<br/>

## ⚙️ Run Locally<br/>

```bash
pip install -r requirements.txt
streamlit run upload_jd.py

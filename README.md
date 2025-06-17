# 🧠 AI Resume Ranker <br/>

![image](https://github.com/user-attachments/assets/ff5afec3-7e62-4c58-bab1-bc9b837d636e)

![image](https://github.com/user-attachments/assets/a005d52b-6e22-4009-bf57-044bafd0e609)

![image](https://github.com/user-attachments/assets/2caf5827-e12f-4ddc-8fdb-34b953466ce9)


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
